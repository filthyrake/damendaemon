"""Independent Qwen3.5 still-image encoder for experimental serving.

FP16 projection storage, FP32 normalization/residuals/attention and outputs.
CUDA projections reuse dllm's compensated Tensor Core products. The published
encoder equations and checkpoint layout are validated against Transformers,
which is never imported here.
"""
from .config import read_config
from math import isqrt
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors import safe_open

from .qwen2 import weight_files


class VisionWeights:
    def __init__(self, path, *, projection_dtype=torch.float16):
        if projection_dtype not in (torch.float16,torch.float32):
            raise ValueError('unsupported vision projection storage')
        self.projection_dtype=projection_dtype
        self.path = Path(path)
        config = read_config(self.path/'config.json')
        self.config = c = config.get('vision_config',{})
        expected = dict(depth=12,hidden_size=768,intermediate_size=3072,num_heads=12,
                        out_hidden_size=1024,patch_size=16,spatial_merge_size=2,
                        temporal_patch_size=2,in_channels=3,num_position_embeddings=2304,
                        hidden_act='gelu_pytorch_tanh')
        if (config.get('model_type')!='qwen3_5' or any(c.get(k)!=v for k,v in expected.items())
                or c.get('deepstack_visual_indexes')):
            raise ValueError('unsupported Qwen3.5 vision geometry')
        rope = c.get('rope_parameters')
        if rope is not None and (rope.get('rope_type')!='axial' or rope.get('rope_theta')!=10000):
            raise ValueError('unsupported vision rotary configuration')
        h,inner,merged = c['hidden_size'],c['intermediate_size'],c['hidden_size']*4
        schema = {'patch_embed.proj.weight':(h,3,2,16,16),'patch_embed.proj.bias':(h,),
                  'pos_embed.weight':(c['num_position_embeddings'],h),
                  'merger.norm.weight':(h,),'merger.norm.bias':(h,),
                  'merger.linear_fc1.weight':(merged,merged),'merger.linear_fc1.bias':(merged,),
                  'merger.linear_fc2.weight':(c['out_hidden_size'],merged),
                  'merger.linear_fc2.bias':(c['out_hidden_size'],)}
        for layer in range(c['depth']):
            root=f'blocks.{layer}.'
            for name in ('norm1','norm2'):
                schema[root+name+'.weight']=schema[root+name+'.bias']=(h,)
            for name,outputs,inputs in (('attn.qkv',3*h,h),('attn.proj',h,h),
                                        ('mlp.linear_fc1',inner,h),('mlp.linear_fc2',h,inner)):
                schema[root+name+'.weight']=(outputs,inputs)
                schema[root+name+'.bias']=(outputs,)
        self.weights={}
        for filename in weight_files(self.path):
            with safe_open(str(filename),framework='pt',device='cpu') as reader:
                for key in reader.keys():
                    if not key.startswith('model.visual.'):continue
                    name=key[len('model.visual.'):]
                    if name not in schema or tuple(reader.get_slice(key).get_shape())!=schema[name]:
                        raise ValueError(f'unsupported vision tensor: {name}')
                    if name in self.weights:raise ValueError(f'duplicate vision tensor: {name}')
                    projection=name.endswith('.weight') and name!='pos_embed.weight' and '.norm' not in name
                    value=reader.get_tensor(key).to(projection_dtype if projection else torch.float32)
                    if name=='patch_embed.proj.weight':value=value.flatten(1)
                    self.weights[name]=value.contiguous()
        if self.weights.keys()!=schema.keys():raise ValueError('incomplete vision checkpoint')
        self.bytes=sum(value.numel()*value.element_size() for value in self.weights.values())


class VisionEncoder:
    def __init__(self, model, device='cuda', *, kernels='auto', max_patches=4096, attention='sdpa'):
        if kernels not in ('auto','torch') or type(max_patches) is not int or max_patches<4:
            raise ValueError('invalid vision execution policy')
        self.model,self.c=model,model.config
        self.device=torch.device(device)
        self.max_patches=max_patches
        if attention not in ('sdpa','eager'):
            raise ValueError('invalid vision attention policy')
        self.attention=attention
        self.w={key:value.to(self.device) for key,value in model.weights.items()}
        self.product=None
        if self.device.type=='cuda' and kernels=='auto' and model.projection_dtype==torch.float16:
            from .precise_matmul import linear
            self.product=linear
        self.backend='compensated-cuda-vision' if self.product else 'torch-fp32-vision'
        self.bytes=sum(value.numel()*value.element_size() for value in self.w.values())

    def linear(self, value, name):
        weight=self.w[name+'.weight']
        result=(self.product(value.contiguous(),weight) if self.product
                else F.linear(value,weight.float()))
        return result+self.w[name+'.bias']

    def norm(self, value, name):
        return F.layer_norm(value,(value.shape[-1],),self.w[name+'.weight'],self.w[name+'.bias'],1e-6)

    def positions(self, grids):
        coordinates=[]
        for _,height,width in grids:
            index=torch.arange(height*width,device=self.device)
            block=index//4
            row=(block//(width//2))*2+(index//2)%2
            column=(block%(width//2))*2+index%2
            coordinates.append(torch.stack((row,column),dim=-1))
        return coordinates

    def positional_embeddings(self, coordinates, grids):
        side=isqrt(self.c['num_position_embeddings'])
        learned=[]
        for coords,(_,height,width) in zip(coordinates,grids):
            # Scalar divisors avoid a synchronous host tensor upload during capture.
            scaled=coords.float()*(side-1)
            point=torch.stack((scaled[:,0]/(height-1),scaled[:,1]/(width-1)),dim=-1)
            lower=point.floor().long();upper=(lower+1).clamp_max(side-1)
            fraction=point-lower
            rows=torch.stack((lower[:,0],upper[:,0]),dim=-1)
            columns=torch.stack((lower[:,1],upper[:,1]),dim=-1)
            row_weights=torch.stack((1-fraction[:,0],fraction[:,0]),dim=-1)
            column_weights=torch.stack((1-fraction[:,1],fraction[:,1]),dim=-1)
            indices=(rows[:,:,None]*side+columns[:,None,:]).reshape(-1,4)
            coefficients=(row_weights[:,:,None]*column_weights[:,None,:]).reshape(-1,4)
            learned.append((self.w['pos_embed.weight'][indices]*coefficients[:,:,None]).sum(1))
        coords=torch.cat(coordinates).float()
        head_dim=self.c['hidden_size']//self.c['num_heads']
        frequencies=10000.**(-torch.arange(0,head_dim//2,2,device=self.device).float()/(head_dim//2))
        angles=(coords[:,:,None]*frequencies).flatten(1)
        angles=torch.cat((angles,angles),dim=-1)
        return torch.cat(learned),angles.cos()[:,None,:],angles.sin()[:,None,:]

    @torch.inference_mode()
    def encode(self, pixel_values, grids, *, trace=None):
        if (not grids or any(len(grid)!=3 or any(type(v) is not int for v in grid)
                or grid[0]!=1 or min(grid[1:])<2 or grid[1]%2 or grid[2]%2 for grid in grids)):
            raise ValueError('still-image grids must be (1, even height, even width)')
        lengths=[height*width for _,height,width in grids]
        total=sum(lengths)
        if total>self.max_patches:raise ValueError('vision patch budget exceeded')
        if pixel_values.shape!=(total,1536) or pixel_values.dtype!=torch.float32:
            raise ValueError('expected FP32 packed RGB temporal patches')
        pixels=pixel_values.to(self.device)
        def record(name,value):
            if trace is not None:trace[name]=value.detach().cpu()
        value=self.linear(pixels,'patch_embed.proj')
        record('patch',value)
        position,cos,sin=self.positional_embeddings(self.positions(grids),grids)
        value=value+position
        record('positioned',value)
        heads=self.c['num_heads'];dim=self.c['hidden_size']//heads
        for layer in range(self.c['depth']):
            root=f'blocks.{layer}.'
            q,k,v=self.linear(self.norm(value,root+'norm1'),root+'attn.qkv').reshape(total,3,heads,dim).unbind(1)
            def rotate(x):
                left,right=x.chunk(2,dim=-1)
                return x*cos+torch.cat((-right,left),dim=-1)*sin
            q,k=rotate(q),rotate(k)
            attended=[]
            offset=0
            for length in lengths:
                chunk=slice(offset,offset+length)
                query,key,values=(x[chunk].transpose(0,1).unsqueeze(0) for x in (q,k,v))
                if self.attention=='eager':
                    scores=(query@key.transpose(-1,-2))*(dim**-.5)
                    output=scores.softmax(-1)@values
                else:
                    output=F.scaled_dot_product_attention(query,key,values,is_causal=False)
                attended.append(output.squeeze(0).transpose(0,1).reshape(length,-1))
                offset+=length
            value=value+self.linear(torch.cat(attended),root+'attn.proj')
            hidden=self.linear(self.norm(value,root+'norm2'),root+'mlp.linear_fc1')
            value=value+self.linear(F.gelu(hidden,approximate='tanh'),root+'mlp.linear_fc2')
            record(f'block.{layer}',value)
        normalized=self.norm(value,'merger.norm').reshape(-1,self.c['hidden_size']*4)
        merged=self.linear(F.gelu(self.linear(normalized,'merger.linear_fc1')),'merger.linear_fc2')
        record('merged',merged)
        return merged

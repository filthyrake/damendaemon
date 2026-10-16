"""Independent FP32 Mamba-1 loader and slot-owned state-space executor."""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from .config import eos_ids, read_config, _positive_number
from .device import selected_device,available_memory,empty_cache
from .batch import BatchExecutor


def config(path):
    c=read_config(Path(path)/'config.json')
    from .tokenizer_metadata import read_generation_config
    generation=Path(path)/'generation_config.json'
    if generation.exists():
        saved=read_generation_config(generation)
        if 'eos_token_id' in saved:c['eos_token_id']=saved['eos_token_id']
    if (c.get('model_type')!='mamba' or c.get('architectures')!=['MambaForCausalLM']
            or c.get('quantization_config') or c.get('hidden_act','silu')!='silu'
            or c.get('rms_norm',True) is not True or c.get('residual_in_fp32',True) is not True
            or c.get('ssm_cfg')):
        raise ValueError('unsupported Mamba-1 configuration')
    for key in ('hidden_size','intermediate_size','num_hidden_layers','state_size','conv_kernel','vocab_size','time_step_rank'):
        if type(c.get(key)) is not int or c[key]<=0:raise ValueError('invalid Mamba dimension: '+key)
    if c['state_size']>32 or c['conv_kernel']>16:raise ValueError('unsupported Mamba state or convolution width')
    for key,default in (('use_bias',False),('use_conv_bias',True),('tie_word_embeddings',True)):
        c.setdefault(key,default)
        if type(c[key]) is not bool:raise ValueError('invalid Mamba bias/tie policy')
    eps=c.get('layer_norm_epsilon',1.e-5)
    if not _positive_number(eps):raise ValueError('invalid Mamba normalization epsilon')
    if not eos_ids(c.get('eos_token_id')) or any(t>=c['vocab_size'] for t in eos_ids(c['eos_token_id'])):
        raise ValueError('Mamba requires valid EOS tokens')
    # Bound initial qualification; absence of rotary positions is not a claim
    # that arbitrary context lengths have been validated.
    c['max_position_embeddings']=2048;c['layer_norm_epsilon']=eps
    return c


def packable(name,shape):
    name=name.removesuffix('.weight').rsplit('.',1)[-1]
    return len(shape)==2 and shape[1]%32==0 and name in ('head','lm_head','in_proj','out_proj','x_proj','dt_proj')


def weight_bytes(path,quantization=None):
    from .quantization import mode
    from .qwen2 import weight_files
    quantization=mode(quantization)
    if quantization not in ('none','q8_0'):raise ValueError('Mamba supports none or q8_0 weight precision')
    c=config(path);total=0;has_head=False
    for file in weight_files(path):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            for name in reader.keys():
                shape=reader.get_slice(name).get_shape();count=math.prod(shape)
                total+=count+count//32*2 if quantization=='q8_0' and packable(name,shape) else count*4
                has_head=has_head or name=='lm_head.weight'
    if quantization=='q8_0' and c['tie_word_embeddings'] and not has_head and c['hidden_size']%32==0:
        # Quantized output head is separate from the unchanged input embedding.
        count=c['vocab_size']*c['hidden_size'];total+=count+count//32*2
    return total


class MambaWeights:
    task='generation';quantization='none';weight_cache_status='disabled'

    def __init__(self,path,max_context,quantization=None):
        from .quantization import mode
        from .qwen2 import weight_files
        self.quantization=mode(quantization)
        if self.quantization not in ('none','q8_0'):raise ValueError('Mamba supports none or q8_0 weight precision')
        self.path=Path(path).resolve();self.config=c=config(path)
        if type(max_context) is not int or not 1<=max_context<=2048:raise ValueError('Mamba context must be 1..2048')
        self.max_context=max_context;self.eos=eos_ids(c['eos_token_id'])
        self.tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.tokenizer.no_padding();self.tokenizer.no_truncation()
        state={}
        for file in weight_files(path):
            with safe_open(str(file),framework='pt',device='cpu') as reader:
                for name in reader.keys():state[name]=reader.get_tensor(name).float()
        def take(name,shape):
            value=state.pop(name)
            if tuple(value.shape)!=shape:raise ValueError('Mamba checkpoint shape mismatch: '+name)
            return value.contiguous()
        h,i,s,r=c['hidden_size'],c['intermediate_size'],c['state_size'],c['time_step_rank']
        w=self.weights={'embed':take('backbone.embeddings.weight',(c['vocab_size'],h)),
            'norm':take('backbone.norm_f.weight',(h,))}
        if c['tie_word_embeddings']:
            w['head']=w['embed']
            if 'lm_head.weight' in state and not torch.equal(take('lm_head.weight',w['embed'].shape),w['head']):
                raise ValueError('Mamba tied output weights disagree')
        else:w['head']=take('lm_head.weight',(c['vocab_size'],h))
        for layer in range(c['num_hidden_layers']):
            prefix=f'backbone.layers.{layer}.';target=f'{layer}.'
            w[target+'norm']=take(prefix+'norm.weight',(h,));prefix+='mixer.'
            for name,shape in (('in_proj',(2*i,h)),('x_proj',(r+2*s,i)),('dt_proj',(i,r)),('out_proj',(h,i))):
                w[target+name]=take(prefix+name+'.weight',shape)
            w[target+'dt_bias']=take(prefix+'dt_proj.bias',(i,))
            w[target+'conv']=take(prefix+'conv1d.weight',(i,1,c['conv_kernel'])).squeeze(1)
            w[target+'conv_bias']=take(prefix+'conv1d.bias',(i,)) if c['use_conv_bias'] else torch.zeros(i)
            w[target+'A']=-take(prefix+'A_log',(i,s)).exp()
            w[target+'D']=take(prefix+'D',(i,))
            if c['use_bias']:
                w[target+'in_bias']=take(prefix+'in_proj.bias',(2*i,))
                w[target+'out_bias']=take(prefix+'out_proj.bias',(h,))
        if state:raise ValueError('unexpected Mamba checkpoint tensors: '+', '.join(state))
        self.scales={}
        if self.quantization=='q8_0':
            from .quantization import quantize
            for name,value in list(w.items()):
                if packable(name,value.shape):w[name],self.scales[name]=quantize(value)
        tensors=list(w.values())+list(self.scales.values())
        self.bytes=sum(x.numel()*x.element_size() for x in {id(x):x for x in tensors}.values())

    def raw_tokens(self,prompt):
        tokens=self.tokenizer.encode(prompt,add_special_tokens=True).ids if isinstance(prompt,str) else list(prompt)
        if not tokens or any(type(t) is not int or not 0<=t<self.config['vocab_size'] for t in tokens):
            raise ValueError('invalid Mamba prompt tokens')
        return tokens

    def chat_tokens(self,*args,**kwargs):
        raise ValueError('Mamba has no chat template; use raw completions')


class MambaExecutor:
    supports_chunked_prefill=True
    prefix_can_rewind=False
    capture_graphs=BatchExecutor.capture_graphs
    decode_batch=BatchExecutor.decode_batch

    def __init__(self,model,use_graph=True,max_batch=4,kernels=None):
        import os
        if not 1<=max_batch<=16:raise ValueError('batch capacity must be 1..16')
        self.model=model;self.c=c=model.config;self.context=model.max_context
        self.device=selected_device();self.max_batch=max_batch
        self.compute_dtype=self.residual_dtype=torch.float32
        selected=kernels or os.environ.get('DLLM_KERNELS','auto')
        fuse=os.environ.get('DLLM_MAMBA_FUSED_DT','0')
        if fuse not in ('0','1'):raise ValueError('Mamba time-step fusion must be 0 or 1')
        self.fuse_time_step=fuse=='1'
        if selected not in ('auto','torch','triton','metal'):raise ValueError('invalid Mamba kernel policy')
        self.kernels=self.recurrent=None
        if selected!='torch':
            if self.device=='cuda':
                if selected=='metal':raise ValueError('Metal requires Apple Silicon')
                from . import kernels as native
                from . import mamba_kernels as recurrent
            else:
                if selected=='triton':raise ValueError('Triton requires CUDA')
                from . import metal_kernels as native
                from . import mamba_metal as recurrent
            self.kernels=native;self.recurrent=recurrent
        empty_cache()
        conv_dim=c.get('conv_dim',c['intermediate_size'])
        state_bytes=c['num_hidden_layers']*max_batch*(c['intermediate_size']*c['state_size']+conv_dim*c['conv_kernel'])*4
        if model.bytes+state_bytes+512*1024**2>available_memory():raise MemoryError('insufficient GPU memory for Mamba')
        copies={};self.w={}
        for name,value in model.weights.items():
            if id(value) not in copies:copies[id(value)]=value.to(self.device)
            self.w[name]=copies[id(value)]
        if model.scales:
            from .quantization import Linear
            self.linear=Linear(self.w,model.scales,self.device,self.linear)
        self.conv=[torch.zeros((max_batch,conv_dim,c['conv_kernel']),device=self.device) for _ in range(c['num_hidden_layers'])]
        self.state=[torch.zeros((max_batch,c['intermediate_size'],c['state_size']),device=self.device) for _ in range(c['num_hidden_layers'])]
        self.lengths=[0]*max_batch;self.length=0;self.batch_graphs={}
        self.kernel_backend=('triton' if self.device=='cuda' else 'metal') if self.kernels else 'torch'
        if use_graph and self.device=='cuda':self.capture_graphs()

    def linear(self,x,weight,bias=None):
        if self.kernels and self.device=='mps':return self.kernels.linear(x,weight,bias)
        if self.kernels and x.shape[0]==1:
            y=self.kernels.linear_fp32(x,weight)
            return y if bias is None else y+bias
        return F.linear(x,weight,bias)

    def norm(self,x,weight):
        eps=self.c['layer_norm_epsilon']
        if self.kernels:return self.kernels.rms(x,weight,eps,True)
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight

    def tensor_recurrence(self,x,gate,params,dt,layer,slots,reset):
        w=self.w;p=f'{layer}.';s=self.c['state_size'];r=self.c['time_step_rank']
        b,c=params.split((r,s,s),-1)[1:]
        state=torch.zeros_like(self.state[layer].index_select(0,slots)) if reset else self.state[layer].index_select(0,slots)
        output=[]
        dt=F.softplus(dt+w[p+'dt_bias'])
        for t in range(x.shape[1]):
            step=dt[:,t,:,None]
            state=torch.exp(step*w[p+'A'])*state+(step*b[:,t,None,:])*x[:,t,:,None]
            y=(state*c[:,t,None,:]).sum(-1)+w[p+'D']*x[:,t]
            output.append(y*F.silu(gate[:,t]))
        self.state[layer].index_copy_(0,slots,state)
        return torch.stack(output,1)

    def run(self,ids,slots,reset,return_logits=True):
        batch,length=ids.shape;i=self.c['intermediate_size'];r=self.c['time_step_rank']
        x=F.embedding(ids,self.w['embed']).reshape(batch*length,-1)
        for layer in range(self.c['num_hidden_layers']):
            p=f'{layer}.'
            projected=self.linear(self.norm(x,self.w[p+'norm']),self.w[p+'in_proj'],self.w.get(p+'in_bias'))
            if self.recurrent:
                # Both kernels read the packed projection directly, avoiding
                # two materialized signal/gate copies per layer and token.
                signal=gate=projected.reshape(batch,length,2*i)
                signal=self.recurrent.conv(signal,self.w[p+'conv'],self.w[p+'conv_bias'],self.conv[layer],slots,reset)
            else:
                signal,gate=[v.reshape(batch,length,i).contiguous() for v in projected.chunk(2,-1)]
                if reset:
                    self.conv[layer].index_copy_(0,slots,F.pad(signal.transpose(1,2),(self.c['conv_kernel']-length,0)))
                    signal=F.silu(F.conv1d(signal.transpose(1,2),self.w[p+'conv'][:,None],self.w[p+'conv_bias'],padding=self.c['conv_kernel']-1,groups=i)[:,:,:length]).transpose(1,2).contiguous()
                else:
                    history=self.conv[layer].index_select(0,slots)
                    combined=torch.cat((history,signal.transpose(1,2)),-1)
                    self.conv[layer].index_copy_(0,slots,combined[:,:,-self.c['conv_kernel']:])
                    signal=F.silu(F.conv1d(combined[:,:,1:],self.w[p+'conv'][:,None],self.w[p+'conv_bias'],groups=i)).transpose(1,2).contiguous()
            params=self.linear(signal.reshape(batch*length,i),self.w[p+'x_proj']).reshape(batch,length,-1)
            fused_dt=self.fuse_time_step and self.recurrent is not None and self.device=='mps' and length==1
            dt=None if fused_dt else self.linear(params[:,:,:r].contiguous().reshape(batch*length,r),self.w[p+'dt_proj']).reshape(batch,length,i)
            if self.recurrent:
                output=self.recurrent.scan(signal,gate,params,dt,self.w[p+'dt_bias'],self.w[p+'A'],self.w[p+'D'],self.state[layer],slots,reset,
                    **({'dt_weight':self.w[p+'dt_proj']} if fused_dt else {}))
            else:output=self.tensor_recurrence(signal,gate,params,dt,layer,slots,reset)
            x=x+self.linear(output.reshape(batch*length,i),self.w[p+'out_proj'],self.w.get(p+'out_bias'))
        if not return_logits:return None
        x=x.reshape(batch,length,-1)[:,-1].contiguous()
        return self.linear(self.norm(x,self.w['norm']),self.w['head'])

    @torch.inference_mode()
    def forward_batch(self,ids,positions,slots):return self.run(ids[:,None],slots,False)

    @torch.inference_mode()
    def prefill_chunk(self,tokens,slot,start,final=True):
        if (type(slot) is not int or not 0<=slot<self.max_batch
                or type(start) is not int or start<0
                or (start and self.lengths[slot]!=start)
                or not tokens or start+len(tokens)>self.context):
            raise ValueError('invalid Mamba partial prompt state')
        logits=self.run(torch.tensor([tokens],device=self.device),
            torch.tensor([slot],device=self.device),start==0,return_logits=final)
        self.lengths[slot]=start+len(tokens)
        return logits

    def prefix_state_bytes(self,count):
        return sum(t[0].numel()*t.element_size() for t in self.conv+self.state)

    def snapshot_prefix(self,slot,count):
        if self.lengths[slot]!=count:raise ValueError('Mamba snapshots cannot rewind recurrent state')
        return [t[slot].clone() for t in self.conv+self.state]

    def restore_prefix(self,slot,count,snapshot):
        for target,saved in zip(self.conv+self.state,snapshot):target[slot].copy_(saved)

    @torch.inference_mode()
    def prefill_batch(self,prompts,slots):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(type(s) is not int or not 0<=s<self.max_batch for s in slots)
                or any(not p or len(p)>self.context for p in prompts)):
            raise ValueError('invalid Mamba prompt batch')
        # Equal lengths share projection work without padding advancing state.
        # Bound temporary token activations to one configured context and retain
        # the caller's row order even when lengths are interleaved.
        groups={};outputs=[None]*len(prompts)
        for row,(prompt,slot) in enumerate(zip(prompts,slots)):
            groups.setdefault(len(prompt),[]).append((row,prompt,slot))
        for length,entries in groups.items():
            capacity=max(1,self.context//length)
            for start in range(0,len(entries),capacity):
                chunk=entries[start:start+capacity]
                ids=torch.tensor([p for _,p,_ in chunk],device=self.device)
                assigned=torch.tensor([s for _,_,s in chunk],device=self.device)
                logits=self.run(ids,assigned,True)
                for index,(row,_,slot) in enumerate(chunk):
                    outputs[row]=logits[index:index+1];self.lengths[slot]=length
        return torch.cat(outputs)

    def prefill(self,tokens,slot=0):
        result=self.prefill_batch([tokens],[slot]);self.length=len(tokens);return result

    def decode(self,token):
        result=self.decode_batch([token],[0]);self.length=self.lengths[0];return result

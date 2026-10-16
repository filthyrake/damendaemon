"""Independent single-group Mamba-2 FP32 loading and GPU execution."""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from .config import eos_ids, read_config, _positive_number
from .mamba import MambaWeights,MambaExecutor,packable
from .device import selected_device


def config(path):
    c=read_config(Path(path)/'config.json')
    from .tokenizer_metadata import read_generation_config
    generation=Path(path)/'generation_config.json'
    if generation.exists():
        saved=read_generation_config(generation)
        if 'eos_token_id' in saved:c['eos_token_id']=saved['eos_token_id']
    if (c.get('model_type')!='mamba2' or c.get('architectures')!=['Mamba2ForCausalLM']
            or c.get('quantization_config') or type(c.get('n_groups')) is not int or c.get('n_groups')!=1
            or c.get('hidden_act')!='silu' or c.get('rms_norm') is not True
            or c.get('residual_in_fp32') is not True or c.get('expand')!=2
            or c.get('time_step_limit',[0,float('inf')])!=[0,float('inf')]):
        raise ValueError('unsupported Mamba-2 configuration')
    for name in ('hidden_size','num_heads','head_dim','state_size','conv_kernel','vocab_size','num_hidden_layers'):
        if type(c.get(name)) is not int or c[name]<=0:raise ValueError('invalid Mamba-2 dimension: '+name)
    if c['num_heads']*c['head_dim']!=2*c['hidden_size'] or c['state_size']>256 or c['conv_kernel']>16:
        raise ValueError('unsupported Mamba-2 recurrence geometry')
    for name in ('use_bias','use_conv_bias','tie_word_embeddings'):
        if type(c.get(name)) is not bool:raise ValueError('invalid Mamba-2 bias/tie policy')
    eps=c.get('layer_norm_epsilon')
    if not _positive_number(eps):raise ValueError('invalid Mamba-2 epsilon')
    if not eos_ids(c.get('eos_token_id')) or max(eos_ids(c['eos_token_id']))>=c['vocab_size']:
        raise ValueError('Mamba-2 requires valid EOS tokens')
    c['intermediate_size']=2*c['hidden_size'];c['conv_dim']=c['intermediate_size']+2*c['state_size']
    c['max_position_embeddings']=2048
    return c


def weight_bytes(path,quantization=None):
    from .quantization import mode
    from .qwen2 import weight_files
    quantization=mode(quantization)
    if quantization not in ('none','q8_0'):raise ValueError('Mamba-2 supports none or q8_0 weight precision')
    c=config(path);total=0;has_head=False
    for file in weight_files(path):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            for name in reader.keys():
                shape=reader.get_slice(name).get_shape();count=math.prod(shape)
                total+=count+count//32*2 if quantization=='q8_0' and packable(name,shape) else count*4
                has_head=has_head or name=='lm_head.weight'
    if quantization=='q8_0' and c['tie_word_embeddings'] and not has_head and c['hidden_size']%32==0:
        count=c['vocab_size']*c['hidden_size'];total+=count+count//32*2
    return total


class Mamba2Weights(MambaWeights):
    def __init__(self,path,max_context,quantization=None):
        from .quantization import mode
        from .qwen2 import weight_files
        self.quantization=mode(quantization)
        if self.quantization not in ('none','q8_0'):raise ValueError('Mamba-2 supports none or q8_0 weight precision')
        self.path=Path(path).resolve();self.config=c=config(path)
        if type(max_context) is not int or not 1<=max_context<=2048:raise ValueError('Mamba-2 context must be 1..2048')
        self.max_context=max_context;self.eos=eos_ids(c['eos_token_id']);self.scales={}
        self.tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'))
        self.tokenizer.no_padding();self.tokenizer.no_truncation()
        state={}
        for file in weight_files(path):
            with safe_open(str(file),framework='pt',device='cpu') as reader:
                for name in reader.keys():state[name]=reader.get_tensor(name).float()
        def take(name,shape):
            value=state.pop(name)
            if tuple(value.shape)!=shape:raise ValueError('Mamba-2 checkpoint shape mismatch: '+name)
            return value.contiguous()
        h,i,n,cd=c['hidden_size'],c['intermediate_size'],c['num_heads'],c['conv_dim']
        w=self.weights={'embed':take('backbone.embeddings.weight',(c['vocab_size'],h)),
            'norm':take('backbone.norm_f.weight',(h,))}
        if c['tie_word_embeddings']:
            w['head']=w['embed']
            if 'lm_head.weight' in state and not torch.equal(take('lm_head.weight',w['head'].shape),w['head']):
                raise ValueError('Mamba-2 tied output weights disagree')
        else:w['head']=take('lm_head.weight',(c['vocab_size'],h))
        for layer in range(c['num_hidden_layers']):
            prefix=f'backbone.layers.{layer}.';target=f'{layer}.'
            w[target+'norm']=take(prefix+'norm.weight',(h,));prefix+='mixer.'
            w[target+'in_proj']=take(prefix+'in_proj.weight',(i+cd+n,h))
            w[target+'out_proj']=take(prefix+'out_proj.weight',(h,i))
            w[target+'gated_norm']=take(prefix+'norm.weight',(i,))
            w[target+'conv']=take(prefix+'conv1d.weight',(cd,1,c['conv_kernel'])).squeeze(1)
            w[target+'conv_bias']=take(prefix+'conv1d.bias',(cd,)) if c['use_conv_bias'] else torch.zeros(cd)
            w[target+'A']=-take(prefix+'A_log',(n,)).exp()
            w[target+'D']=take(prefix+'D',(n,));w[target+'dt_bias']=take(prefix+'dt_bias',(n,))
            if c['use_bias']:
                w[target+'in_bias']=take(prefix+'in_proj.bias',(i+cd+n,))
                w[target+'out_bias']=take(prefix+'out_proj.bias',(h,))
        if state:raise ValueError('unexpected Mamba-2 tensors: '+', '.join(state))
        if self.quantization=='q8_0':
            from .quantization import quantize
            for name,value in list(w.items()):
                if packable(name,value.shape):w[name],self.scales[name]=quantize(value)
        tensors=list(w.values())+list(self.scales.values())
        self.bytes=sum(v.numel()*v.element_size() for v in {id(v):v for v in tensors}.values())


class Mamba2Executor(MambaExecutor):
    def __init__(self,model,*args,**kwargs):
        import os
        projection=os.environ.get('DLLM_MAMBA2_COMPENSATED','0')
        if projection not in ('0','1'):raise ValueError('Mamba-2 compensated projection policy must be 0 or 1')
        self.prefill_compensated=projection=='1'
        if self.prefill_compensated:
            if selected_device()!='cuda' or model.quantization!='none':
                raise ValueError('Mamba-2 compensated projections require CUDA and unquantized weights')
            if (model.config['hidden_size'],model.config['intermediate_size'])!=(768,1536):
                raise ValueError('Mamba-2 compensated projections are qualified for 130M geometry only')
            for name,weight in model.weights.items():
                if name.endswith(('in_proj','out_proj')) and (
                        not torch.isfinite(weight).all() or not torch.equal(weight,weight.half().float())):
                    raise ValueError('Mamba-2 compensated projections require exactly representable FP16 weights')
        policy=os.environ.get('DLLM_MAMBA2_SSD','0')
        if policy not in ('0','1'):raise ValueError('Mamba-2 SSD policy must be 0 or 1')
        self.prefill_ssd=policy=='1'
        if self.prefill_ssd and selected_device()!='cuda':raise ValueError('Mamba-2 SSD experiment requires CUDA')
        if self.prefill_ssd:
            from triton.runtime import driver
            properties=driver.active.utils.get_device_properties(torch.cuda.current_device())
            if torch.cuda.get_device_capability()[0]<8 or properties['max_shared_mem']<101376:
                raise ValueError('Mamba-2 SSD requires Ampere or newer and at least 99 KiB shared memory')
        if selected_device()=='cuda':from . import mamba2_kernels as native
        else:from . import mamba2_metal as native
        self.ssd=native
        super().__init__(model,*args,**kwargs)

    def project(self,x,weight,bias=None):
        if self.prefill_compensated and x.shape[0]>=128:
            from .precise_matmul import linear
            # Preserve resident FP32 weights and the existing decode path.
            # The checked conversion is temporary, bounded to one projection.
            result=linear(x,weight.half())
            return result if bias is None else result+bias
        return self.linear(x,weight,bias)

    def run(self,ids,slots,reset,return_logits=True):
        batch,length=ids.shape;i=self.c['intermediate_size'];cd=self.c['conv_dim'];n=self.c['num_heads']
        x=F.embedding(ids,self.w['embed']).reshape(batch*length,-1)
        for layer in range(self.c['num_hidden_layers']):
            p=f'{layer}.'
            packed=self.project(self.norm(x,self.w[p+'norm']),self.w[p+'in_proj'],self.w.get(p+'in_bias'))
            gate,signal,dt=[v.reshape(batch,length,-1).contiguous() for v in packed.split((i,cd,n),-1)]
            if self.recurrent:
                signal=self.recurrent.conv(signal,self.w[p+'conv'],self.w[p+'conv_bias'],self.conv[layer],slots,reset)
                scan=self.ssd.scan
                if self.prefill_ssd and length>=128 and i==1536 and self.c['head_dim']==64 and self.c['state_size']==128:
                    from .mamba2_ssd import scan
                output=scan(signal,gate,dt,self.w[p+'dt_bias'],self.w[p+'A'],self.w[p+'D'],
                    self.state[layer],slots,self.c['head_dim'],reset)
            else:
                history=torch.zeros_like(self.conv[layer].index_select(0,slots)) if reset else self.conv[layer].index_select(0,slots)
                joined=torch.cat((history,signal.transpose(1,2)),-1)
                self.conv[layer].index_copy_(0,slots,joined[:,:,-self.c['conv_kernel']:])
                signal=F.silu(F.conv1d(joined[:,:,1:],self.w[p+'conv'][:,None],self.w[p+'conv_bias'],groups=cd)).transpose(1,2)
                state=torch.zeros_like(self.state[layer].index_select(0,slots)) if reset else self.state[layer].index_select(0,slots)
                head=torch.arange(i,device=x.device)//self.c['head_dim'];out=[]
                steps=F.softplus(dt+self.w[p+'dt_bias'])
                for t in range(length):
                    value,b,c=signal[:,t].split((i,self.c['state_size'],self.c['state_size']),-1)
                    step=steps[:,t,head,None]
                    state=torch.exp(step*self.w[p+'A'][head,None])*state+(step*b[:,None,:])*value[:,:,None]
                    out.append(((state*c[:,None,:]).sum(-1)+self.w[p+'D'][head]*value)*F.silu(gate[:,t]))
                self.state[layer].index_copy_(0,slots,state);output=torch.stack(out,1)
            output=self.norm(output.reshape(batch*length,i),self.w[p+'gated_norm'])
            x=x+self.project(output,self.w[p+'out_proj'],self.w.get(p+'out_bias'))
        if not return_logits:return None
        return self.linear(self.norm(x.reshape(batch,length,-1)[:,-1].contiguous(),self.w['norm']),self.w['head'])

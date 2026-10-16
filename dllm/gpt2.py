"""Learned-position GPT-2 decoder using shared native matrices and KV slots."""
import torch
from .floating_parameter import require_floating
from torch.nn import functional as F
from .starcoder import StarcoderExecutor


def load_weights(state,config):
    if 'transformer.wte.weight' in state:
        state={key.removeprefix('transformer.'):value for key,value in state.items()}
    h,inter=config['hidden_size'],config['intermediate_size']
    def take(name,shape,transpose=False):
        value=require_floating(state.pop(name),name)
        if tuple(value.shape)!=tuple(shape):raise ValueError('GPT-2 checkpoint shape mismatch: '+name)
        return (value.t() if transpose else value).float().contiguous()
    w={'embed':take('wte.weight',(config['vocab_size'],h)),
       'position':take('wpe.weight',(config['max_position_embeddings'],h))}
    w['head']=w['embed']
    if 'lm_head.weight' in state:
        if not torch.equal(require_floating(state.pop('lm_head.weight'),'lm_head.weight').float(),w['head']):
            raise ValueError('GPT-2 checkpoint declares unequal tied output weights')
    for suffix in ('weight','bias'):
        w['norm' if suffix=='weight' else 'norm.bias']=take('ln_f.'+suffix,(h,))
    for i in range(config['num_hidden_layers']):
        prefix=f'h.{i}.'
        # Saved causal masks are non-parameter buffers; attention constructs its own.
        state.pop(prefix+'attn.bias',None)
        state.pop(prefix+'attn.masked_bias',None)
        for name,source in (('norm1','ln_1'),('norm2','ln_2')):
            w[f'{i}.{name}']=take(prefix+source+'.weight',(h,))
            w[f'{i}.{name}.bias']=take(prefix+source+'.bias',(h,))
        for name,source,shape in (('qkv.weight','attn.c_attn',(h,3*h)),
                                  ('out','attn.c_proj',(h,h)),('fc','mlp.c_fc',(h,inter)),
                                  ('down','mlp.c_proj',(inter,h))):
            w[f'{i}.{name}']=take(prefix+source+'.weight',shape,True)
            bias_name='qkv.bias' if name=='qkv.weight' else name+'.bias'
            w[f'{i}.{bias_name}']=take(prefix+source+'.bias',(shape[1],))
    if state:raise ValueError('unexpected GPT-2 checkpoint tensors')
    return w


class Gpt2Executor(StarcoderExecutor):
    def _forward(self,ids,positions,slots,prefill):
        self.linear=self._linear
        count,w=ids.numel(),self.w
        x=F.embedding(ids,w['embed']).to(self.residual_dtype)+F.embedding(positions,w['position']).to(self.residual_dtype)
        for i in range(self.c['num_hidden_layers']):
            packed=self.linear(self.layer_norm(x,f'{i}.norm1'),w[f'{i}.qkv.weight'],w[f'{i}.qkv.bias'])
            q,k,v=[value.reshape(count,self.heads,self.dim) for value in packed.chunk(3,-1)]
            ck,cv=self.kv[i].unbind(0)
            ck[slots,:,positions,:],cv[slots,:,positions,:]=k,v
            # Packed QKV slices have a 3*hidden row stride; native decode
            # attention expects contiguous query rows for heterogeneous batches.
            query=q.contiguous().unsqueeze(2)
            if prefill:
                attended=self.prefill_attention(q.transpose(0,1).unsqueeze(0),
                    ck.index_select(0,slots[:1])[:,:,:count],cv.index_select(0,slots[:1])[:,:,:count])
                attended=attended.transpose(1,2).reshape(count,-1)
            else:
                attended=self.decode_attention(query,ck,cv,positions,slots).reshape(count,-1)
            x=x+self.linear(attended,w[f'{i}.out'],w[f'{i}.out.bias'])
            hidden=self.linear(self.layer_norm(x,f'{i}.norm2'),w[f'{i}.fc'],w[f'{i}.fc.bias'])
            hidden=F.gelu(hidden.float(),approximate='tanh').to(self.compute_dtype)
            x=x+self.linear(hidden,w[f'{i}.down'],w[f'{i}.down.bias'])
        if prefill:x=x[-1:]
        return self._linear(self.layer_norm(x,'norm',torch.float32),w['head'])

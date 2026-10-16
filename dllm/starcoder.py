"""Independent StarCoder2 decoder using shared cache slots and native attention."""
import torch
from torch.nn import functional as F
from .batch import BatchExecutor


def load_weights(state, config):
    h=config['hidden_size'];inter=config['intermediate_size']
    query=config['num_attention_heads']*config['head_dim']
    kv=config['num_key_value_heads']*config['head_dim']
    def take(name,shape):
        if name not in state:raise ValueError('missing StarCoder2 checkpoint tensor: '+name)
        value=state.pop(name)
        if tuple(value.shape)!=shape:
            raise ValueError('StarCoder2 checkpoint shape mismatch: '+name)
        return value.float()
    w = {'embed': take('model.embed_tokens.weight',(config['vocab_size'],h))}
    if config.get('tie_word_embeddings'):
        w['head']=w['embed']
        if 'lm_head.weight' in state and not torch.equal(take('lm_head.weight',(config['vocab_size'],h)),w['head']):
            raise ValueError('StarCoder2 checkpoint declares unequal tied output weights')
    else:w['head']=take('lm_head.weight',(config['vocab_size'],h))
    for suffix in ('weight', 'bias'):
        w['norm' if suffix == 'weight' else 'norm.bias'] = take('model.norm.'+suffix,(h,))
    for i in range(config['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        for name, source in (('norm1','input_layernorm'), ('norm2','post_attention_layernorm')):
            w[f'{i}.{name}'] = take(prefix+source+'.weight',(h,))
            w[f'{i}.{name}.bias'] = take(prefix+source+'.bias',(h,))
        for suffix in (('weight','bias') if config['use_bias'] else ('weight',)):
            w[f'{i}.qkv.{suffix}'] = torch.cat([take(prefix+f'self_attn.{p}_proj.{suffix}',
                (width,h) if suffix=='weight' else (width,)) for p,width in (('q',query),('k',kv),('v',kv))])
            for name, source, shape in (('out','self_attn.o_proj',(h,query)),
                                        ('fc','mlp.c_fc',(inter,h)),('down','mlp.c_proj',(h,inter))):
                w[f'{i}.{name}' + ('.bias' if suffix == 'bias' else '')] = take(prefix+source+'.'+suffix,
                    shape if suffix=='weight' else (shape[0],))
    if state:raise ValueError('unexpected StarCoder2 checkpoint tensors')
    return w


class StarcoderExecutor(BatchExecutor):
    def _linear(self, x, weight, bias=None):
        if weight.dtype == torch.int8:
            return self.quant_linear(x,weight,bias)
        if self.device == 'mps' and self.kernels:
            return self.kernels.linear(x,weight,bias)
        if self.device == 'cuda' and self.kernels and x.shape[0] <= 4:
            result = self.kernels.linear_fp32(x.float(),weight)
            return result if bias is None else result+bias
        return F.linear(x.float(),weight.float(),None if bias is None else bias.float())

    def layer_norm(self, x, name, output_dtype=None):
        return F.layer_norm(x.float(), (x.shape[-1],), self.w[name].float(),
                            self.w[name+'.bias'].float(), self.eps).to(output_dtype or self.compute_dtype)

    @torch.inference_mode()
    def forward(self, ids, positions, prefill, slot=0):
        return self._forward(ids, positions, torch.full_like(positions, slot), prefill)

    @torch.inference_mode()
    def forward_batch(self, ids, positions, slots):
        return self._forward(ids, positions, slots, False)

    def _forward(self, ids, positions, slots, prefill):
        self.linear = self._linear
        count, w = ids.numel(), self.w
        x = F.embedding(ids, w['embed']).to(self.residual_dtype)
        for i in range(self.c['num_hidden_layers']):
            packed = self.linear(self.layer_norm(x, f'{i}.norm1'), w[f'{i}.qkv.weight'], w.get(f'{i}.qkv.bias'))
            ck, cv = self.kv[i].unbind(0)
            if self.kernels:
                query = self.kernels.rope_batch(packed, self.cos, self.sin, positions, slots,
                    ck, cv, self.heads, precise=self.rotary_precise)
            else:
                q, k, v = packed.split([self.heads*self.dim, self.kv_heads*self.dim, self.kv_heads*self.dim], -1)
                cos, sin = self.cos[positions,None,:], self.sin[positions,None,:]
                query = self.rotate(q.view(count,self.heads,self.dim),cos,sin).unsqueeze(2)
                key = self.rotate(k.view(count,self.kv_heads,self.dim),cos,sin)
                ck[slots,:,positions,:], cv[slots,:,positions,:] = key, v.view(count,self.kv_heads,self.dim)
            if prefill:
                # One sequence here; decode uses shared batched graph execution.
                slot = slots[0:1]
                attended = self.prefill_attention(query[:,:,0,:].transpose(0,1).unsqueeze(0),
                    ck.index_select(0,slot)[:,:,:count], cv.index_select(0,slot)[:,:,:count])
                attended = attended.transpose(1,2).reshape(count,-1)
            else:
                attended = self.decode_attention(query,ck,cv,positions,slots).reshape(count,-1)
            x = x + self.linear(attended,w[f'{i}.out'],w.get(f'{i}.out.bias'))
            hidden = self.linear(self.layer_norm(x,f'{i}.norm2'),w[f'{i}.fc'],w.get(f'{i}.fc.bias'))
            hidden = F.gelu(hidden.float(),approximate='tanh').to(self.compute_dtype)
            x = x + self.linear(hidden,w[f'{i}.down'],w.get(f'{i}.down.bias'))
        if prefill:
            x = x[-1:]
        normalized = self.layer_norm(x,'norm',torch.float32)
        return self._linear(normalized,w['head'])

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(slot<0 or slot>=self.max_batch for slot in slots)
                or any(not ids or len(ids)>self.context for ids in prompts)):
            raise ValueError('invalid prefill batch')
        return torch.cat([self.prefill(ids,slot) for ids,slot in zip(prompts,slots)])

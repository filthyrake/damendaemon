"""Independent GPT-NeoX decoder with partial rotary and parallel residuals."""
import torch
from torch.nn import functional as F
from .starcoder import StarcoderExecutor


def load_weights(state, config):
    h, inter = config['hidden_size'], config['intermediate_size']
    heads, dim = config['num_attention_heads'], config['head_dim']
    def take(name, shape):
        value = state.pop(name)
        if tuple(value.shape) != tuple(shape):
            raise ValueError('GPT-NeoX checkpoint shape mismatch: '+name)
        return value.float().contiguous()
    w = {'embed':take('gpt_neox.embed_in.weight', (config['vocab_size'],h)),
         'head':take('embed_out.weight', (config['vocab_size'],h))}
    for suffix in ('weight','bias'):
        w['norm' if suffix=='weight' else 'norm.bias'] = take('gpt_neox.final_layer_norm.'+suffix,(h,))
    for i in range(config['num_hidden_layers']):
        prefix = f'gpt_neox.layers.{i}.'
        for buffer in ('attention.bias','attention.masked_bias','attention.rotary_emb.inv_freq'):
            state.pop(prefix+buffer,None)
        for name,source in (('norm1','input_layernorm'),('norm2','post_attention_layernorm')):
            w[f'{i}.{name}'] = take(prefix+source+'.weight',(h,))
            w[f'{i}.{name}.bias'] = take(prefix+source+'.bias',(h,))
        # Checkpoint rows are head-major Q/K/V; execution uses whole Q/K/V projections.
        packed = take(prefix+'attention.query_key_value.weight',(3*h,h))
        w[f'{i}.qkv.weight'] = packed.reshape(heads,3,dim,h).transpose(0,1).reshape(3*h,h).contiguous()
        bias = take(prefix+'attention.query_key_value.bias',(3*h,))
        w[f'{i}.qkv.bias'] = bias.reshape(heads,3,dim).transpose(0,1).reshape(3*h).contiguous()
        for name,source,shape in (('out','attention.dense',(h,h)),
                                  ('fc','mlp.dense_h_to_4h',(inter,h)),
                                  ('down','mlp.dense_4h_to_h',(h,inter))):
            w[f'{i}.{name}'] = take(prefix+source+'.weight',shape)
            w[f'{i}.{name}.bias'] = take(prefix+source+'.bias',(shape[0],))
    if state:
        raise ValueError('unexpected GPT-NeoX checkpoint tensors')
    return w


class NeoxExecutor(StarcoderExecutor):
    def partial_rotate(self, x, positions):
        width = self.c['rotary_dim']
        rotary, tail = x[...,:width], x[...,width:]
        cos, sin = self.cos[positions,None,:], self.sin[positions,None,:]
        left, right = rotary.chunk(2,-1)
        rotated = rotary*cos + torch.cat((-right,left),-1)*sin
        return torch.cat((rotated,tail),-1)

    def _forward(self, ids, positions, slots, prefill):
        count, w = ids.numel(), self.w
        x = F.embedding(ids,w['embed']).to(self.residual_dtype)
        for i in range(self.c['num_hidden_layers']):
            packed = self._linear(self.layer_norm(x,f'{i}.norm1'),w[f'{i}.qkv.weight'],w[f'{i}.qkv.bias'])
            ck,cv = self.kv[i].unbind(0)
            if self.device=='cuda' and self.kernels:
                query=self.kernels.rope_batch(packed,self.cos,self.sin,positions,slots,
                    ck,cv,self.heads,precise=True)
            else:
                q,k,v = [part.reshape(count,self.heads,self.dim) for part in packed.chunk(3,-1)]
                query = self.partial_rotate(q,positions).contiguous().unsqueeze(2)
                key = self.partial_rotate(k,positions)
                ck[slots,:,positions,:],cv[slots,:,positions,:] = key,v
            if prefill:
                attended = self.prefill_attention(query[:,:,0,:].transpose(0,1).unsqueeze(0),
                    ck.index_select(0,slots[:1])[:,:,:count],cv.index_select(0,slots[:1])[:,:,:count])
                attended = attended.transpose(1,2).reshape(count,-1)
            else:
                attended = self.decode_attention(query,ck,cv,positions,slots).reshape(count,-1)
            attended = self._linear(attended,w[f'{i}.out'],w[f'{i}.out.bias'])
            mlp_input = x if self.c['use_parallel_residual'] else x+attended
            hidden = self._linear(self.layer_norm(mlp_input,f'{i}.norm2'),w[f'{i}.fc'],w[f'{i}.fc.bias'])
            hidden = F.gelu(hidden,approximate='none')
            hidden = self._linear(hidden,w[f'{i}.down'],w[f'{i}.down.bias'])
            x = (hidden+attended)+x if self.c['use_parallel_residual'] else hidden+mlp_input
        if prefill:
            x = x[-1:]
        return self._linear(self.layer_norm(x,'norm',torch.float32),w['head'])

"""Independent OLMoE execution with GPU-resident decode routing."""
import os
import torch
from torch.nn import functional as F
from .batch import BatchExecutor
from .device import selected_device


def load_weights(state, config):
    def matrix(name):
        return state.pop(name).half()
    def norm(name):
        return state.pop(name).float()
    w = {'embed':matrix('model.embed_tokens.weight'),'norm':norm('model.norm.weight')}
    w['head'] = w['embed'] if config.get('tie_word_embeddings') else matrix('lm_head.weight')
    for i in range(config['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        w[f'{i}.qkv.weight'] = torch.cat([matrix(prefix+f'self_attn.{p}_proj.weight') for p in ('q','k','v')])
        w[f'{i}.out'] = matrix(prefix+'self_attn.o_proj.weight')
        for target,source in (('norm1','input_layernorm'),('norm2','post_attention_layernorm'),
                              ('q_norm','self_attn.q_norm'),('k_norm','self_attn.k_norm')):
            w[f'{i}.{target}'] = norm(prefix+source+'.weight')
        w[f'{i}.router'] = matrix(prefix+'mlp.gate.weight')
        up,down = [],[]
        for expert in range(config['num_experts']):
            root = prefix+f'mlp.experts.{expert}.'
            up.append(torch.cat([matrix(root+p+'_proj.weight') for p in ('gate','up')]))
            down.append(matrix(root+'down_proj.weight'))
        # Flatten expert/output axes so the existing Q8 loader keeps row groups.
        w[f'{i}.experts_up'] = torch.cat(up)
        w[f'{i}.experts_down'] = torch.cat(down)
    return w


class OlmoeExecutor(BatchExecutor):
    def __init__(self, model, use_graph=True, max_batch=4, kernels=None):
        if selected_device() != 'cuda' or (kernels or os.environ.get('DLLM_KERNELS','auto')) == 'torch':
            raise ValueError('OLMoE currently requires native CUDA kernels')
        from . import moe_kernels
        self.expert_kernels = moe_kernels
        super().__init__(model,use_graph,max_batch,kernels)
        self.kernel_backend += '-olmoe'

    def _linear(self, x, weight, bias=None):
        if weight.dtype == torch.int8:
            return self.quant_linear(x,weight,bias)
        if x.shape[0] <= 4:
            result = self.kernels.linear_fp32(x,weight)
            return result if bias is None else result+bias
        return F.linear(x,weight.float(),bias)

    def _expert_scales(self, weight):
        return self.quant_linear.scales[id(weight)] if weight.dtype == torch.int8 else None

    def _expert_linear(self, x, weight, expert, outputs):
        matrix = weight.view(self.c['num_experts'],outputs,-1)[expert]
        scales = self._expert_scales(weight)
        if scales is not None:
            from .quantized_kernels import linear
            return linear(x,matrix,scales.view(self.c['num_experts'],outputs,-1)[expert])
        return self._linear(x,matrix)

    def experts(self, hidden, layer, prefill):
        width = self.c['intermediate_size']
        probabilities,selected = self._linear(hidden,self.w[f'{layer}.router']).softmax(-1).topk(self.c['num_experts_per_tok'],dim=-1)
        if self.c['norm_topk_prob']:
            probabilities = probabilities/probabilities.sum(-1,keepdim=True)
        up,down = self.w[f'{layer}.experts_up'],self.w[f'{layer}.experts_down']
        if prefill:
            # Dynamic grouping is confined to prefill. Decode uses only device
            # routing below, including during CUDA graph capture/replay.
            result = torch.zeros_like(hidden)
            for expert in range(self.c['num_experts']):
                tokens,routes = torch.where(selected==expert)
                if tokens.numel()==0:
                    continue
                packed = self._expert_linear(hidden.index_select(0,tokens),up,expert,2*width)
                activated = self.kernels.swiglu(packed,True)
                projected = self._expert_linear(activated,down,expert,self.c['hidden_size'])
                result.index_add_(0,tokens,projected*probabilities[tokens,routes,None])
            return result
        packed = self.expert_kernels.routed_linear(hidden,up,self._expert_scales(up),selected,2*width,True)
        activated = self.kernels.swiglu(packed,True)
        projected = self.expert_kernels.routed_linear(activated,down,self._expert_scales(down),selected,self.c['hidden_size'],False)
        return self.expert_kernels.combine(projected,probabilities)

    @torch.inference_mode()
    def forward(self, ids, positions, prefill, slot=0):
        return self._forward(ids,positions,torch.full_like(positions,slot),prefill)

    @torch.inference_mode()
    def forward_batch(self, ids, positions, slots):
        return self._forward(ids,positions,slots,False)

    def _forward(self, ids, positions, slots, prefill):
        count,w = ids.numel(),self.w
        x = F.embedding(ids,w['embed']).float()
        for i in range(self.c['num_hidden_layers']):
            packed = self._linear(self.norm(x,w[f'{i}.norm1']),w[f'{i}.qkv.weight'])
            q,k,v = packed.split([self.heads*self.dim,self.kv_heads*self.dim,self.kv_heads*self.dim],-1)
            q,k = self.norm(q.contiguous(),w[f'{i}.q_norm']),self.norm(k.contiguous(),w[f'{i}.k_norm'])
            clip = self.c.get('clip_qkv')
            if clip is not None:
                q,k,v = q.clamp(-clip,clip),k.clamp(-clip,clip),v.clamp(-clip,clip)
            packed = torch.cat((q,k,v),dim=-1)
            ck,cv = self.kv[i].unbind(0)
            query = self.kernels.rope_batch(packed,self.cos,self.sin,positions,slots,ck,cv,self.heads,precise=True)
            if prefill:
                slot = slots[:1]
                attended = self.prefill_attention(query[:,:,0,:].transpose(0,1).unsqueeze(0),
                    ck.index_select(0,slot)[:,:,:count],cv.index_select(0,slot)[:,:,:count])
                attended = attended.transpose(1,2).reshape(count,-1)
            else:
                attended = self.decode_attention(query,ck,cv,positions,slots).reshape(count,-1)
            x = x+self._linear(attended,w[f'{i}.out'])
            x = x+self.experts(self.norm(x,w[f'{i}.norm2']),i,prefill)
        if prefill:
            x=x[-1:]
        return self._linear(self.norm(x,w['norm']),w['head'])

    @torch.inference_mode()
    def prefill_batch(self, prompts, slots):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(slot<0 or slot>=self.max_batch for slot in slots)
                or any(not ids or len(ids)>self.context for ids in prompts)):
            raise ValueError('invalid prefill batch')
        return torch.cat([self.prefill(ids,slot) for ids,slot in zip(prompts,slots)])

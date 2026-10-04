"""Independent BLOOM decoder and ALiBi attention (Press et al., arXiv:2108.12409)."""
import torch
from torch.nn import functional as F
from .device import selected_device
from .starcoder import StarcoderExecutor


def alibi_slopes(heads, device='cpu'):
    """Geometric slopes, followed by interleaved slopes for non-power-of-two heads."""
    power = 1 << (heads.bit_length() - 1)
    base = torch.tensor(2 ** (-8 / power), device=device, dtype=torch.float32)
    slopes = base ** torch.arange(1, power + 1, device=device)
    if heads != power:
        extra = torch.tensor(2 ** (-4 / power), device=device, dtype=torch.float32)
        slopes = torch.cat((slopes, extra ** torch.arange(1, 2 * (heads-power), 2, device=device)))
    return slopes


def load_weights(state, config):
    if 'transformer.word_embeddings.weight' in state:
        state = {key.removeprefix('transformer.'): value for key, value in state.items()}
    h, heads, d = config['hidden_size'], config['num_attention_heads'], config['head_dim']
    def take(name, shape):
        value = state.pop(name)
        if tuple(value.shape) != shape:
            raise ValueError('BLOOM checkpoint shape mismatch: ' + name)
        return value.float().contiguous()
    w = {'embed': take('word_embeddings.weight', (config['vocab_size'], h))}
    w['head'] = w['embed']
    if 'lm_head.weight' in state and not torch.equal(state.pop('lm_head.weight').float(), w['head']):
        raise ValueError('BLOOM checkpoint declares unequal tied output weights')
    for name, source in (('embed_norm', 'word_embeddings_layernorm'), ('norm', 'ln_f')):
        w[name] = take(source + '.weight', (h,))
        w[name + '.bias'] = take(source + '.bias', (h,))
    for i in range(config['num_hidden_layers']):
        prefix = f'h.{i}.'
        for name, source in (('norm1', 'input_layernorm'), ('norm2', 'post_attention_layernorm')):
            w[f'{i}.{name}'] = take(prefix + source + '.weight', (h,))
            w[f'{i}.{name}.bias'] = take(prefix + source + '.bias', (h,))
        packed = take(prefix + 'self_attention.query_key_value.weight', (3*h, h))
        w[f'{i}.qkv.weight'] = packed.reshape(heads, 3, d, h).transpose(0, 1).reshape(3*h, h).contiguous()
        bias = take(prefix + 'self_attention.query_key_value.bias', (3*h,))
        w[f'{i}.qkv.bias'] = bias.reshape(heads, 3, d).transpose(0, 1).reshape(3*h).contiguous()
        for name, source, shape in (('out', 'self_attention.dense', (h,h)),
                                   ('fc', 'mlp.dense_h_to_4h', (4*h,h)),
                                   ('down', 'mlp.dense_4h_to_h', (h,4*h))):
            w[f'{i}.{name}'] = take(prefix + source + '.weight', shape)
            w[f'{i}.{name}.bias'] = take(prefix + source + '.bias', (shape[0],))
    if state:
        raise ValueError('unexpected BLOOM checkpoint tensors')
    return w


class BloomExecutor(StarcoderExecutor):
    def __init__(self, model, *args, **kwargs):
        # The parent warms up and captures decode during construction.
        self.alibi_slopes = alibi_slopes(model.config['num_attention_heads'], selected_device())
        self.alibi_times = torch.arange(model.max_context, device=selected_device())
        self.alibi_bias = self.alibi_slopes[:, None] * self.alibi_times[None, :]
        super().__init__(model, *args, **kwargs)

    def alibi_attention(self, query, ck, cv, positions, slots, prefill):
        if not prefill and self.device == 'cuda' and self.kernels:
            return self.kernels.attention(query, ck, cv, positions, slots, alibi=self.alibi_slopes)
        if prefill:
            count = positions.numel()
            keys = ck.index_select(0, slots[:1])[:, :, :count]
            values = cv.index_select(0, slots[:1])[:, :, :count]
            # Bound additive mask workspace rather than materializing H*L*L.
            outputs = []
            for start in range(0, count, 128):
                stop = min(count, start + 128)
                visible = self.alibi_times[:count][None, :] <= positions[start:stop, None]
                mask = self.alibi_bias[:, None, :count].masked_fill(~visible[None, :, :], -float('inf'))
                outputs.append(F.scaled_dot_product_attention(query[:, :, start:stop], keys, values, attn_mask=mask))
            return torch.cat(outputs, dim=2)
        keys, values = ck.index_select(0, slots), cv.index_select(0, slots)
        visible = self.alibi_times[None, :] <= positions[:, None]
        mask = self.alibi_bias[None, :, None, :].masked_fill(~visible[:, None, None, :], -float('inf'))
        return F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)

    def _forward(self, ids, positions, slots, prefill):
        count, w = ids.numel(), self.w
        x = self.layer_norm(F.embedding(ids, w['embed']), 'embed_norm')
        post_norm = self.c['apply_residual_connection_post_layernorm']
        for i in range(self.c['num_hidden_layers']):
            normalized = self.layer_norm(x, f'{i}.norm1')
            residual = normalized if post_norm else x
            packed = self._linear(normalized, w[f'{i}.qkv.weight'], w[f'{i}.qkv.bias'])
            q, k, v = [part.reshape(count, self.heads, self.dim) for part in packed.chunk(3, -1)]
            ck, cv = self.kv[i].unbind(0)
            ck[slots, :, positions, :], cv[slots, :, positions, :] = k, v
            query = q.transpose(0, 1).unsqueeze(0) if prefill else q.contiguous().unsqueeze(2)
            attended = self.alibi_attention(query, ck, cv, positions, slots, prefill)
            attended = attended.transpose(1, 2).reshape(count, -1) if prefill else attended.reshape(count, -1)
            x = residual + self._linear(attended, w[f'{i}.out'], w[f'{i}.out.bias'])
            normalized = self.layer_norm(x, f'{i}.norm2')
            residual = normalized if post_norm else x
            hidden = self._linear(normalized, w[f'{i}.fc'], w[f'{i}.fc.bias'])
            hidden = F.gelu(hidden, approximate='tanh')
            x = residual + self._linear(hidden, w[f'{i}.down'], w[f'{i}.down.bias'])
        if prefill:
            x = x[-1:]
        return self._linear(self.layer_norm(x, 'norm'), w['head'])

"""Consecutive real-weight text-path executor for Qwen3.5-MoE (#14, cell 4).

Runs layers 0..3 of the real Qwen3.6-35B-A3B text path end to end (embedding,
three gated-deltanet linear-attention layers, one full-attention layer with
partial rotary + qk-norm + output gate, MoE FFN per layer, final norm) over
bounded safetensors slice reads from the read-only snapshot (BF16->FP32, no
staging copy), and compares two independent implementations over the SAME
real tensors:

  ours: the qualified dense-hybrid math (damendaemon/hybrid.py attention and
        linear-attention methods, ported to the qwen3_5_moe tensor names,
        including the upstream zero-centered (1.0 + w) RMSNorm convention and
        the plain-weight gated norm) composed with our HybridMoeFeedForward,
  upstream: the eager Transformers 5.20.2 Qwen3_5MoeDecoderLayer stack
        (_attn_implementation='eager', _experts_implementation='eager').

The probe is a fixed-seed synthetic normal matrix, NOT real activations.
Full-attention and norm ops are op-for-op identical (expected near-bitwise);
linear attention compares our sequential recurrence against the upstream
chunked kernel, so float-tolerance agreement is the expectation there and
metrics are recorded exactly as measured. No gate is relaxed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward

LINEAR_NAMES = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a')
FULL_NAMES = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


def digest(tensor):
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def read_tensors(base, names):
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    opened = {}
    out = {}
    for name in names:
        shard = weight_map.get(name)
        if shard is None:
            raise ValueError(f'missing checkpoint tensor {name}')
        if shard not in opened:
            opened[shard] = safe_open(base / shard, framework='pt')
        raw = opened[shard].get_tensor(name)
        if raw.dtype is not torch.bfloat16:
            raise ValueError(f'unexpected dtype {raw.dtype} for {name}')
        out[name] = raw.float()
    return out


def layer_names(layer, kind):
    root = f'model.language_model.layers.{layer}.'
    names = [root + 'input_layernorm.weight', root + 'post_attention_layernorm.weight']
    names += [f'{root}mlp.{n}' for n in (
        'gate.weight', 'experts.gate_up_proj', 'experts.down_proj',
        'shared_expert.gate_proj.weight', 'shared_expert.up_proj.weight',
        'shared_expert.down_proj.weight', 'shared_expert_gate.weight')]
    if kind == 'linear_attention':
        names += [f'{root}linear_attn.{n}.weight' for n in LINEAR_NAMES]
        names += [f'{root}linear_attn.out_proj.weight', f'{root}linear_attn.conv1d.weight',
                  f'{root}linear_attn.dt_bias', f'{root}linear_attn.A_log', f'{root}linear_attn.norm.weight']
    else:
        names += [f'{root}self_attn.{n}.weight' for n in FULL_NAMES]
        names += [f'{root}self_attn.q_norm.weight', f'{root}self_attn.k_norm.weight']
    return names


def rms(x, weight, eps):
    # Upstream Qwen3_5MoeRMSNorm: zero-centered weight convention.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def ours_linear(w, root, x, c):
    b, t, _ = x.shape
    lk, lv, dk, dv = c['linear_num_key_heads'], c['linear_num_value_heads'], \
        c['linear_key_head_dim'], c['linear_value_head_dim']
    width, kernel = 2 * lk * dk + lv * dv, c['linear_conv_kernel_dim']
    qkv = F.linear(x, w[root + 'linear_attn.in_proj_qkv.weight']).transpose(1, 2)
    convolved = F.conv1d(qkv, w[root + 'linear_attn.conv1d.weight'], padding=kernel - 1,
                         groups=width)[..., :t]
    qkv = F.silu(convolved).transpose(1, 2)
    q, k, v = qkv.split((lk * dk, lk * dk, lv * dv), -1)
    q, k = [a.reshape(b, t, lk, dk) for a in (q, k)]
    v = v.reshape(b, t, lv, dv)
    beta = F.linear(x, w[root + 'linear_attn.in_proj_b.weight']).sigmoid()
    decay = -w[root + 'linear_attn.A_log'].exp() * F.softplus(
        F.linear(x, w[root + 'linear_attn.in_proj_a.weight']) + w[root + 'linear_attn.dt_bias'])
    z = F.linear(x, w[root + 'linear_attn.in_proj_z.weight']).reshape(b, t, lv, dv)
    q, k = [l2norm(a) for a in (q, k)]
    q, k = [a.repeat_interleave(lv // lk, dim=2) for a in (q, k)]
    q = q * (dk ** -0.5)
    state = torch.zeros(b, lv, dk, dv, dtype=torch.float32)
    outputs = []
    for step in range(t):
        state = state * decay[:, step][..., None, None].exp()
        delta = (v[:, step] - (state * k[:, step, :, :, None]).sum(-2)) * beta[:, step, :, None]
        state = state + k[:, step, :, :, None] * delta.unsqueeze(-2)
        outputs.append((state * q[:, step, :, :, None]).sum(-2))
    out = torch.stack(outputs, 1)
    # Upstream RMSNormGated: plain weight (not the zero-centered convention), silu(z) gate.
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + c['rms_norm_eps']) \
        * w[root + 'linear_attn.norm.weight']
    out = out * F.silu(z)
    return F.linear(out.reshape(b, t, lv * dv), w[root + 'linear_attn.out_proj.weight'])


def ours_full(w, root, x, positions, cos, sin, c):
    b, t, _ = x.shape
    heads, kh, dim, rd = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim'], c['rotary_dim']
    q = F.linear(x, w[root + 'self_attn.q_proj.weight']).view(b, t, heads, 2 * dim)
    q, gate = q.chunk(2, -1)
    gate = gate.reshape(b, t, heads * dim)
    k = F.linear(x, w[root + 'self_attn.k_proj.weight']).view(b, t, kh, dim)
    v = F.linear(x, w[root + 'self_attn.v_proj.weight']).view(b, t, kh, dim)
    eps = c['rms_norm_eps']
    q = rms(q, w[root + 'self_attn.q_norm.weight'], eps)
    k = rms(k, w[root + 'self_attn.k_norm.weight'], eps)

    def rotate(a):
        rotated, rest = a[..., :rd], a[..., rd:]
        left, right = rotated.chunk(2, -1)
        return torch.cat((rotated * cos + torch.cat((-right, left), -1) * sin, rest), -1)

    q, k = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2)
    v = v.transpose(1, 2).repeat_interleave(heads // kh, 1)
    scale = dim ** -0.5
    scores = torch.matmul(q, k.repeat_interleave(heads // kh, 1).transpose(2, 3)) * scale
    causal = torch.full((t, t), float('-inf')).triu(1)
    weights = torch.softmax(scores + causal, dim=-1, dtype=torch.float32)
    out = torch.matmul(weights, v).transpose(1, 2).reshape(b, t, heads * dim)
    out = out * gate.sigmoid()
    return F.linear(out, w[root + 'self_attn.o_proj.weight'])


def ours_ffn(w, root, c):
    return HybridMoeFeedForward(
        w[root + 'mlp.gate.weight'], w[root + 'mlp.experts.gate_up_proj'], w[root + 'mlp.experts.down_proj'],
        w[root + 'mlp.shared_expert.gate_proj.weight'], w[root + 'mlp.shared_expert.up_proj.weight'],
        w[root + 'mlp.shared_expert.down_proj.weight'], w[root + 'mlp.shared_expert_gate.weight'],
        top_k=c['num_experts_per_tok'], scoring='softmax', renormalize=True,
        workspace_bytes=2 ** 40, max_output_bytes=2 ** 40)


def metrics(a, b):
    diff = (a - b).abs()
    scale = b.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {'max_abs': float(diff.max()), 'rms_rel': float((diff / scale).pow(2).mean().sqrt())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--output', default='docs/evidence/architectures/qwen35-moe-text-executor-layers0-3.json')
    args = parser.parse_args()

    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    c = config['text_config']
    c['rotary_dim'] = int(c['head_dim'] * c['rope_parameters']['partial_rotary_factor'])

    names = ['model.language_model.embed_tokens.weight', 'model.language_model.norm.weight']
    for layer in range(args.layers):
        names += layer_names(layer, c['layer_types'][layer])
    w = read_tensors(base, names)
    digests = {}
    for name, tensor in w.items():
        if 'mlp.experts' in name:
            continue  # large fused expert tensors hashed below with short keys
        short = ('embedding' if 'embed' in name else 'final_norm' if name.endswith('norm.weight')
                 and 'layers' not in name else name.split('layers.')[1].replace('.weight', ''))
        digests[short] = digest(tensor)
    # Hash the two large fused expert tensors per layer from their FP32 bytes.
    for layer in range(args.layers):
        for suffix in ('mlp.experts.gate_up_proj', 'mlp.experts.down_proj'):
            name = f'model.language_model.layers.{layer}.{suffix}'
            digests[f'layer{layer}.{suffix.split(".")[-1]}'] = hashlib.sha256(
                w[name].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    generator = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, c['vocab_size'], (1, args.tokens), generator=generator)
    hidden = w['model.language_model.embed_tokens.weight'][ids]

    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding)
    from types import SimpleNamespace
    import transformers

    ns = SimpleNamespace(
        num_experts=c['num_experts'], num_experts_per_tok=c['num_experts_per_tok'],
        hidden_size=c['hidden_size'], moe_intermediate_size=c['moe_intermediate_size'],
        shared_expert_intermediate_size=c['shared_expert_intermediate_size'],
        hidden_act=c['hidden_act'], layer_types=c['layer_types'],
        num_attention_heads=c['num_attention_heads'], num_key_value_heads=c['num_key_value_heads'],
        head_dim=c['head_dim'], attention_bias=c['attention_bias'],
        attention_dropout=0.0, rms_norm_eps=c['rms_norm_eps'],
        linear_num_key_heads=c['linear_num_key_heads'], linear_num_value_heads=c['linear_num_value_heads'],
        linear_key_head_dim=c['linear_key_head_dim'], linear_value_head_dim=c['linear_value_head_dim'],
        linear_conv_kernel_dim=c['linear_conv_kernel_dim'],
        max_position_embeddings=c['max_position_embeddings'],
        rope_parameters=c['rope_parameters'], rope_scaling=None, _experts_implementation='eager',
        _attn_implementation='eager')

    rotary = Qwen3_5MoeTextRotaryEmbedding(ns)
    position_ids = torch.arange(args.tokens).view(1, -1)
    cos, sin = rotary(hidden, position_ids)
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)

    upstream_layers = []
    for layer in range(args.layers):
        block = Qwen3_5MoeDecoderLayer(ns, layer)
        root = f'model.language_model.layers.{layer}.'
        with torch.no_grad():
            block.input_layernorm.weight = torch.nn.Parameter(w[root + 'input_layernorm.weight'],
                                                              requires_grad=False)
            block.post_attention_layernorm.weight = torch.nn.Parameter(
                w[root + 'post_attention_layernorm.weight'], requires_grad=False)
            for attr, name in (('gate.weight', 'mlp.gate.weight'),
                               ('experts.gate_up_proj', 'mlp.experts.gate_up_proj'),
                               ('experts.down_proj', 'mlp.experts.down_proj'),
                               ('shared_expert.gate_proj.weight', 'mlp.shared_expert.gate_proj.weight'),
                               ('shared_expert.up_proj.weight', 'mlp.shared_expert.up_proj.weight'),
                               ('shared_expert.down_proj.weight', 'mlp.shared_expert.down_proj.weight'),
                               ('shared_expert_gate.weight', 'mlp.shared_expert_gate.weight')):
                target = block.mlp
                for part in attr.split('.')[:-1]:
                    target = getattr(target, part)
                setattr(target, attr.split('.')[-1],
                        torch.nn.Parameter(w[root + name], requires_grad=False))
            if c['layer_types'][layer] == 'full_attention':
                for proj in FULL_NAMES:
                    getattr(block.self_attn, proj).weight = torch.nn.Parameter(
                        w[root + f'self_attn.{proj}.weight'], requires_grad=False)
                block.self_attn.q_norm.weight = torch.nn.Parameter(
                    w[root + 'self_attn.q_norm.weight'], requires_grad=False)
                block.self_attn.k_norm.weight = torch.nn.Parameter(
                    w[root + 'self_attn.k_norm.weight'], requires_grad=False)
            else:
                for proj in LINEAR_NAMES:
                    getattr(block.linear_attn, proj).weight = torch.nn.Parameter(
                        w[root + f'linear_attn.{proj}.weight'], requires_grad=False)
                block.linear_attn.out_proj.weight = torch.nn.Parameter(
                    w[root + 'linear_attn.out_proj.weight'], requires_grad=False)
                block.linear_attn.conv1d.weight = torch.nn.Parameter(
                    w[root + 'linear_attn.conv1d.weight'], requires_grad=False)
                block.linear_attn.dt_bias = torch.nn.Parameter(w[root + 'linear_attn.dt_bias'],
                                                               requires_grad=False)
                block.linear_attn.A_log = torch.nn.Parameter(w[root + 'linear_attn.A_log'],
                                                             requires_grad=False)
                block.linear_attn.norm.weight = torch.nn.Parameter(
                    w[root + 'linear_attn.norm.weight'], requires_grad=False)
            block.eval()
        upstream_layers.append(block)

    causal_mask = torch.full((1, 1, args.tokens, args.tokens), float('-inf')).triu(1)
    ours, theirs = hidden, hidden.clone()
    per_layer = []
    with torch.no_grad():
        for layer in range(args.layers):
            kind = c['layer_types'][layer]
            root = f'model.language_model.layers.{layer}.'
            eps = c['rms_norm_eps']
            # ours
            normed = rms(ours, w[root + 'input_layernorm.weight'], eps)
            mixer = (ours_full(w, root, normed, position_ids, cos, sin, c) if kind == 'full_attention'
                     else ours_linear(w, root, normed, c))
            ours = ours + mixer
            ours = ours + ours_ffn(w, root, c).forward(rms(ours, w[root + 'post_attention_layernorm.weight'], eps))
            # upstream
            normed_t = theirs
            theirs = upstream_layers[layer](hidden_states=normed_t,
                                            position_embeddings=rotary(normed_t, position_ids),
                                            attention_mask=causal_mask if kind == 'full_attention' else None,
                                            position_ids=position_ids)
            per_layer.append({'layer': layer, 'kind': kind, 'ours_vs_upstream': metrics(ours, theirs)})
    final_ours = rms(ours, w['model.language_model.norm.weight'], c['rms_norm_eps'])
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    final_norm = Qwen3_5MoeRMSNorm(c['hidden_size'], eps=c['rms_norm_eps'])
    final_norm.weight = torch.nn.Parameter(w['model.language_model.norm.weight'], requires_grad=False)
    with torch.no_grad():
        final_theirs = final_norm(theirs)

    report = {
        'snapshot': str(base), 'layers': args.layers, 'tokens': args.tokens,
        'implementation': 'ours: qualified dense-hybrid math ported to qwen3_5_moe names + HybridMoeFeedForward',
        'upstream': {'transformers_version': transformers.__version__,
                     'reference': 'Qwen3_5MoeDecoderLayer eager + Qwen3_5MoeTextRotaryEmbedding'},
        'config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'tensor_sha256': digests,
        'probe': {'kind': 'synthetic random token IDs (not real text)', 'seed': args.seed,
                  'shape': [1, args.tokens, c['hidden_size']]},
        'per_layer': per_layer,
        'final_hidden': metrics(final_ours, final_theirs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'per_layer': [{'layer': p['layer'], 'kind': p['kind'],
                                    **p['ours_vs_upstream']} for p in per_layer],
                     'final_hidden': report['final_hidden']}, indent=1))


if __name__ == '__main__':
    main()

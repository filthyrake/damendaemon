"""Single-layer real-weight executor for the Qwen3.5-MoE text FFN (#14).

Reads one layer's MoE FFN tensors from the real Qwen3.6-35B-A3B snapshot by
bounded safetensors slice reads (no full materialization, no writes to the
read-only NAS), converts BF16 to FP32 host memory (~3.5 GiB peak), and runs
three computations over the SAME real tensors:

  1. our ``HybridMoeFeedForward`` (qualified softmax + renormalize policy),
  2. an independent per-token naive control (plain torch ops),
  3. the eager upstream reference ``Qwen3_5MoeSparseMoeBlock`` from
     Transformers 5.20.2 ``models/qwen3_5_moe/modeling_qwen3_5_moe.py``
     (skippable with --skip-upstream where transformers>=5 is unavailable).

The probe is a fixed-seed synthetic normal matrix, NOT real activations.
Agreement of implementations on real weights is the claim; model quality is
not. Output JSON pins the config and per-tensor digests, the probe seed,
selected expert IDs, and the deviation metrics. No gate is relaxed: metrics
are recorded exactly as measured.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward

MLP_NAMES = (
    'mlp.gate.weight',
    'mlp.experts.gate_up_proj',
    'mlp.experts.down_proj',
    'mlp.shared_expert.gate_proj.weight',
    'mlp.shared_expert.up_proj.weight',
    'mlp.shared_expert.down_proj.weight',
    'mlp.shared_expert_gate.weight',
)


def digest(tensor):
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_layer(base, layer, text_config):
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    prefix = f'model.language_model.layers.{layer}.'
    opened = {}
    tensors = {}
    digests = {}
    for suffix in MLP_NAMES:
        name = prefix + suffix
        shard = weight_map.get(name)
        if shard is None:
            raise ValueError(f'layer {layer} has no MoE tensor {name}')
        if shard not in opened:
            opened[shard] = safe_open(base / shard, framework='pt')
        raw = opened[shard].get_tensor(name)
        if raw.dtype is torch.bfloat16:
            digests[name] = digest(raw)
            tensors[suffix] = raw.float()
        else:
            raise ValueError(f'unexpected checkpoint dtype {raw.dtype} for {name}')
    expected = {
        'mlp.gate.weight': (text_config['num_experts'], text_config['hidden_size']),
        'mlp.experts.gate_up_proj': (text_config['num_experts'], 2 * text_config['moe_intermediate_size'],
                                      text_config['hidden_size']),
        'mlp.experts.down_proj': (text_config['num_experts'], text_config['hidden_size'],
                                   text_config['moe_intermediate_size']),
        'mlp.shared_expert.gate_proj.weight': (text_config['shared_expert_intermediate_size'],
                                                text_config['hidden_size']),
        'mlp.shared_expert.up_proj.weight': (text_config['shared_expert_intermediate_size'],
                                              text_config['hidden_size']),
        'mlp.shared_expert.down_proj.weight': (text_config['hidden_size'],
                                                text_config['shared_expert_intermediate_size']),
        'mlp.shared_expert_gate.weight': (1, text_config['hidden_size']),
    }
    for suffix, shape in expected.items():
        if tuple(tensors[suffix].shape) != shape:
            raise ValueError(f'{suffix} shape {tuple(tensors[suffix].shape)} != contract {shape}')
    return tensors, digests


def naive_control(t, hidden, top_k):
    """Independent per-token reference: plain torch ops, no shared code."""
    out = torch.zeros_like(hidden)
    for b in range(hidden.shape[0]):
        for i in range(hidden.shape[1]):
            x = hidden[b, i]
            probs = torch.softmax(F.linear(x, t['mlp.gate.weight']), -1, dtype=torch.float32)
            values, ids = probs.topk(top_k, -1)
            values = values / values.sum()
            routed = torch.zeros_like(x)
            for slot in range(top_k):
                gate, up = F.linear(x, t['mlp.experts.gate_up_proj'][ids[slot]]).chunk(2)
                routed = routed + F.linear(F.silu(gate) * up, t['mlp.experts.down_proj'][ids[slot]]) * values[slot]
            shared = F.linear(F.silu(F.linear(x, t['mlp.shared_expert.gate_proj.weight']))
                              * F.linear(x, t['mlp.shared_expert.up_proj.weight']),
                              t['mlp.shared_expert.down_proj.weight'])
            combine = torch.sigmoid(F.linear(x, t['mlp.shared_expert_gate.weight']))
            out[b, i] = routed + shared * combine
    return out


def upstream_block(t, text_config, hidden):
    from types import SimpleNamespace

    import transformers
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    config = SimpleNamespace(
        num_experts=text_config['num_experts'],
        num_experts_per_tok=text_config['num_experts_per_tok'],
        hidden_size=text_config['hidden_size'],
        moe_intermediate_size=text_config['moe_intermediate_size'],
        shared_expert_intermediate_size=text_config['shared_expert_intermediate_size'],
        hidden_act=text_config['hidden_act'],
        _experts_implementation='eager',
    )
    block = Qwen3_5MoeSparseMoeBlock(config)
    with torch.no_grad():
        # Borrow the real FP32 tensors as Parameters without copying, so our
        # FFN and the upstream reference compute over identical bytes.
        block.gate.weight = torch.nn.Parameter(t['mlp.gate.weight'], requires_grad=False)
        block.experts.gate_up_proj = torch.nn.Parameter(t['mlp.experts.gate_up_proj'], requires_grad=False)
        block.experts.down_proj = torch.nn.Parameter(t['mlp.experts.down_proj'], requires_grad=False)
        block.shared_expert.gate_proj.weight = torch.nn.Parameter(t['mlp.shared_expert.gate_proj.weight'],
                                                                  requires_grad=False)
        block.shared_expert.up_proj.weight = torch.nn.Parameter(t['mlp.shared_expert.up_proj.weight'],
                                                                requires_grad=False)
        block.shared_expert.down_proj.weight = torch.nn.Parameter(t['mlp.shared_expert.down_proj.weight'],
                                                                  requires_grad=False)
        block.shared_expert_gate.weight = torch.nn.Parameter(t['mlp.shared_expert_gate.weight'],
                                                             requires_grad=False)
        block.eval()
        return block(hidden), transformers.__version__


def metrics(a, b):
    diff = (a - b).abs()
    scale = b.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {'max_abs': float(diff.max()), 'rms_rel': float((diff / scale).pow(2).mean().sqrt())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', help='Qwen3.6-35B-A3B checkpoint directory (read-only)')
    parser.add_argument('--layer', type=int, default=3, help='MoE layer index (default 3, full-attention)')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--output', default='docs/evidence/architectures/qwen35-moe-executor-layer3.json')
    parser.add_argument('--skip-upstream', action='store_true',
                        help='skip the Transformers 5 reference comparison')
    args = parser.parse_args()

    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5_moe':
        raise ValueError('not a qwen3_5_moe checkpoint')
    text_config = config['text_config']
    tensors, digests = load_layer(base, args.layer, text_config)

    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.randn(args.batch, args.tokens, text_config['hidden_size'],
                         generator=generator, dtype=torch.float32) * 0.5

    ffn = HybridMoeFeedForward(
        tensors['mlp.gate.weight'], tensors['mlp.experts.gate_up_proj'], tensors['mlp.experts.down_proj'],
        tensors['mlp.shared_expert.gate_proj.weight'], tensors['mlp.shared_expert.up_proj.weight'],
        tensors['mlp.shared_expert.down_proj.weight'], tensors['mlp.shared_expert_gate.weight'],
        top_k=text_config['num_experts_per_tok'], scoring='softmax', renormalize=True)
    ours = ffn.forward(hidden)
    control = naive_control(tensors, hidden, text_config['num_experts_per_tok'])

    from damendaemon.expert_selection import select_experts
    logits = F.linear(hidden.reshape(-1, text_config['hidden_size']), tensors['mlp.gate.weight'])
    weights, selected = select_experts(logits, text_config['num_experts_per_tok'],
                                       scoring='softmax', renormalize=True)

    report = {
        'snapshot': str(base),
        'layer': args.layer,
        'layer_type': config['text_config']['layer_types'][args.layer],
        'policy': {'scoring': 'softmax', 'renormalize': True,
                   'pin': 'Transformers 5.20.2 Qwen3_5MoeTopKRouter (eager)'},
        'config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'tensor_sha256': digests,
        'probe': {'kind': 'synthetic normal (not real activations)', 'seed': args.seed,
                  'shape': [args.batch, args.tokens, text_config['hidden_size']]},
        'routing': {'selected_experts_first_row': selected[0].tolist(),
                    'weight_min': float(weights.min()), 'weight_max': float(weights.max())},
        'ours_vs_control': metrics(ours, control),
    }
    if not args.skip_upstream:
        reference, version = upstream_block(tensors, text_config, hidden)
        report['upstream'] = {'transformers_version': version,
                             'reference': 'Qwen3_5MoeSparseMoeBlock (eager)'}
        report['ours_vs_upstream'] = metrics(ours, reference)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('layer', 'layer_type', 'policy',
                                             'ours_vs_control', 'ours_vs_upstream') if k in report}, indent=1))


main()

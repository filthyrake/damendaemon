"""Real-weight error budget for E2M1/E8M0 expert residency (#14, cell 6).

The routed-expert bank of Qwen3.6-35B-A3B is 32.2B values (~60 GiB BF16
across 40 layers) and exceeds both shared GPUs; E4M3FN residency (~32 GB)
does not fit the 24-GiB A30 either. E2M1 with one E8M0 power-of-two scale
per 32 values (~0.531 byte/value, ~15.8 GiB bank) is the only single-GPU
option. This cell measures the real-weight error budget BEFORE any GPU
wiring, using the audited component simulations
(``damendaemon.activation_simulation``, DeepSeek-V4-Flash conventions) on
real checkpoint tensors:

  arms (routed experts only; router, shared expert and combine gate stay
  FP32 in every arm, which is the residency plan):
    fp8            - E4M3FN weights-only, one scale per 64 values
    fp4            - E2M1 weights-only, one E8M0 scale per 32 values
    fp4_hadamard   - E2M1 after normalized-Hadamard weight rotation:
                     effective weight dequant(W @ H) @ H (H symmetric and
                     orthonormal), identical math to the runtime
                     rotate-input/rotate-back scheme

Simulated quantization: dequantized FP32 tensors are substituted into the
audited HybridMoeFeedForward, so the FP32 op order matches the cell-3
reference exactly and the measured deviation is purely the quantization
effect. Metrics are recorded exactly as measured; nothing is gated here.
Memory figures are arithmetic (bytes per value), labeled as such - not a
measurement. The probe is a fixed-seed synthetic normal matrix, NOT real
activations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from damendaemon.activation_simulation import simulate_fp4, simulate_fp8, normalized_hadamard
from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward
from scripts.experiments.qwen35_moe_text_executor import digest, metrics, read_tensors

MLP_SUFFIXES = (
    'mlp.gate.weight', 'mlp.experts.gate_up_proj', 'mlp.experts.down_proj',
    'mlp.shared_expert.gate_proj.weight', 'mlp.shared_expert.up_proj.weight',
    'mlp.shared_expert.down_proj.weight', 'mlp.shared_expert_gate.weight')
EXPERT_SUFFIXES = ('mlp.experts.gate_up_proj', 'mlp.experts.down_proj')
ARMS = ('fp8', 'fp4', 'fp4_hadamard')
BYTES_PER_VALUE = {'bf16': 2.0, 'e4m3fn': 1.0 + 1 / 64, 'e2m1_e8m0': 0.5 + 1 / 32}


def arm_tensor(tensor, arm, rotations):
    if arm == 'fp8':
        return simulate_fp8(tensor)
    if arm == 'fp4':
        return simulate_fp4(tensor)
    hadamard = rotations.setdefault(tensor.shape[-1], normalized_hadamard(torch.eye(tensor.shape[-1])))
    return simulate_fp4(tensor @ hadamard) @ hadamard


def ffn_for(tensors, root, c):
    return HybridMoeFeedForward(
        tensors[root + 'mlp.gate.weight'], tensors[root + 'mlp.experts.gate_up_proj'],
        tensors[root + 'mlp.experts.down_proj'], tensors[root + 'mlp.shared_expert.gate_proj.weight'],
        tensors[root + 'mlp.shared_expert.up_proj.weight'],
        tensors[root + 'mlp.shared_expert.down_proj.weight'],
        tensors[root + 'mlp.shared_expert_gate.weight'],
        top_k=c['num_experts_per_tok'], scoring='softmax', renormalize=True,
        workspace_bytes=2 ** 40, max_output_bytes=2 ** 40)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--layers', default='0,3,9,20,39')
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--output', default='docs/evidence/architectures/qwen35-moe-expert-fp4-budget.json')
    args = parser.parse_args()

    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    c = config['text_config']
    layers = [int(part) for part in args.layers.split(',')]
    for layer in layers:
        if not 0 <= layer < c['num_hidden_layers']:
            raise ValueError(f'layer {layer} out of range')

    generator = torch.Generator().manual_seed(args.seed)
    probe = torch.randn(2, 6, c['hidden_size'], generator=generator, dtype=torch.float32) * 0.5

    rotations = {}
    per_arm = {arm: {} for arm in ARMS}
    digests = {}
    with torch.inference_mode():
        for layer in layers:
            root = f'model.language_model.layers.{layer}.'
            names = [root + suffix for suffix in MLP_SUFFIXES]
            tensors = read_tensors(base, names)
            for suffix in MLP_SUFFIXES:
                digests[f'layer{layer}.{suffix}'] = digest(tensors[root + suffix])
            reference = ffn_for(tensors, root, c).forward(probe)
            for arm in ARMS:
                effective = dict(tensors)
                for suffix in EXPERT_SUFFIXES:
                    effective[root + suffix] = arm_tensor(tensors[root + suffix], arm, rotations)
                quantized = ffn_for(effective, root, c).forward(probe)
                per_arm[arm][layer] = metrics(quantized, reference)
                del effective, quantized
            del tensors, reference

    values_per_layer = c['num_experts'] * (2 * c['moe_intermediate_size'] * c['hidden_size']
                                           + c['hidden_size'] * c['moe_intermediate_size'])
    bank_values = values_per_layer * c['num_hidden_layers']
    memory = {name: {'bytes_per_value': spec, 'bank_bytes_arithmetic': bank_values * spec}
              for name, spec in BYTES_PER_VALUE.items()}

    report = {
        'snapshot': str(base), 'layers': layers, 'arms': list(ARMS),
        'quantized_scope': 'routed expert bank only; router, shared expert and combine gate stay FP32',
        'simulations': 'damendaemon.activation_simulation (DeepSeek-V4-Flash conventions, pinned in module)',
        'config_sha256': hashlib.sha256((base / 'config.json').read_bytes()).hexdigest(),
        'tensor_sha256': digests,
        'probe': {'kind': 'synthetic normal (not real activations)', 'seed': args.seed,
                  'shape': [2, 6, c['hidden_size']]},
        'per_arm_per_layer': {arm: {str(layer): per_arm[arm][layer] for layer in layers} for arm in ARMS},
        'worst_per_arm': {arm: max(per_arm[arm][layer]['max_abs'] for layer in layers) for arm in ARMS},
        'expert_bank_memory_arithmetic': memory,
        'bank_values': bank_values,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'worst_per_arm': report['worst_per_arm'],
                      'per_arm_per_layer': report['per_arm_per_layer'],
                      'bank_gib': {k: round(v['bank_bytes_arithmetic'] / 2 ** 30, 1)
                                   for k, v in memory.items()}}, indent=1))


if __name__ == '__main__':
    main()

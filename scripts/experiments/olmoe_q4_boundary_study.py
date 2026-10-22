"""CPU router-boundary frequency study on the retained Q4 OLMoE golden trace (#577).

Replays the teacher-forced trace through the HF reference (Q4_0 storage policy,
FP32 arithmetic, CPU) and records, at every layer of every position, the routing
probability gaps around the top-k cutoff: the near-tie distribution that a
sub-ppm arithmetic difference must cross to flip an expert selection.

Usage:
  python scripts/experiments/olmoe_q4_boundary_study.py <trace.pt> <out.json>
"""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, '/tmp/damendaemon-50b')
from reference_model import load_reference  # noqa: E402


@torch.inference_mode()
def main():
    trace_path, output = sys.argv[1], sys.argv[2]
    trace = torch.load(trace_path, map_location='cpu', weights_only=True)
    print('model:', trace['path'], '| complete:', trace.get('complete'))
    model = load_reference(trace['path'], reference_quantization='q4_0', local_files_only=True,
                           trust_remote_code=False, torch_dtype=torch.float32,
                           attn_implementation='eager').eval()
    config = model.config
    top_k = config.num_experts_per_tok
    print('experts:', config.num_experts, '| top_k:', top_k, '| layers:', config.num_hidden_layers)

    thresholds = (1e-8, 1e-7, 1e-6, 1e-5)
    gaps = []  # cutoff gap for every routed row

    hooks = []
    for block in model.model.layers:
        gate = block.mlp.gate

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            probs = output.float().softmax(-1)
            top = probs.topk(top_k + 1, dim=-1)
            gap = (top.values[:, top_k - 1] - top.values[:, top_k]).clamp_min(0)
            gaps.extend(gap.tolist())

        hooks.append(gate.register_forward_hook(hook))
    try:
        for event_index, event in enumerate(trace['trace']):
            if event['kind'] == 'prefill':
                for ids in event['prompts']:
                    model(torch.tensor([ids]), use_cache=False)
            else:
                model(torch.tensor([[t] for t in event['tokens']]), use_cache=False)
            print('event', event_index, event['kind'], 'done', flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    gaps_tensor = torch.tensor(gaps)
    report = {
        'scope': 'router-boundary frequency study (#577); CPU reference, Q4_0 storage policy',
        'model': trace['path'],
        'rows': len(gaps),
        'thresholds': thresholds,
        'near_tie_counts': {str(t): int((gaps_tensor < t).sum()) for t in thresholds},
        'gap_quantiles': {str(q): float(torch.quantile(gaps_tensor.float(), torch.tensor(q)))
                          for q in (0.01, 0.05, 0.25, 0.5)},
        'min_gap': float(gaps_tensor.min()),
        'rows_below_1e-6_per_1e6_rows': float((gaps_tensor < 1e-6).float().mean() * 1e6),
    }
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

"""Plan the two-rank A30+A2 Q4 32K-capable split for Qwen3.8-27B (#535/#536 precheck).

Loads the pinned checkpoint once (Q4_0 packed, CPU) and runs the exact
production planner (`choose_stage_splits`) against two explicit device budgets.
CPU planning only; no device allocation and no performance claim.
"""
import argparse
import json
from pathlib import Path

import torch

from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('context', type=int)
    parser.add_argument('--device0-mib', type=int, default=22528)
    parser.add_argument('--device1-mib', type=int, default=13312)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Planning-only admission: the loader policy pins Q4 to the A30 identity,
    # compensated precision and context <=512 (production contract). The
    # planner never executes — it only packs weights and plans byte budgets —
    # so the identity/context gates are substituted here, the same way the
    # #68/#70 context probe substitutes them for bounded execution.
    import os
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    from damendaemon import hybrid_q4_policy
    original_device_name = hybrid_q4_policy.device_name
    hybrid_q4_policy.device_name = lambda: 'NVIDIA A30'
    original_validate = hybrid_q4_policy.validate

    def validate(config, max_context=None):
        if max_context is not None and not 512 <= max_context <= 32768:
            raise ValueError('Q4 planning is bounded to 512..32768 tokens')
        return original_validate(config, None if max_context is None else 512)

    hybrid_q4_policy.validate = validate
    from damendaemon.qwen2 import ModelWeights
    from damendaemon.hybrid_layout import choose_stage_splits
    model = ModelWeights(args.model, max_context=512, quantization='q4_0')
    hybrid_q4_policy.device_name = original_device_name
    model.max_context = args.context
    budgets = [args.device0_mib*1024**2, args.device1_mib*1024**2]
    placement = choose_stage_splits(model.weights, model.config, model.max_context, 1,
                                    2, budgets, scales=model.scales)
    found = bool(placement['storage_plan_found'])
    stages = list(placement['stages']) if found else []
    report = dict(completed=found, model=str(args.model),
                  context=args.context, budgets_mib=[args.device0_mib, args.device1_mib],
                  quantization='q4_0', batch=1, placement=placement,
                  model_packed_bytes=model.bytes)
    if not found:
        report['reason'] = placement['reason']
    save(args.output, report)
    summary = []
    for stage in stages:
        summary.append(dict(layers=[stage['layer_start'], stage['layer_end']],
                            weight_bytes=stage['weight_bytes'], scale_bytes=stage['scale_bytes'],
                            state_bytes=stage['state_bytes'],
                            required_mib=round(stage['required_memory_bytes']/1024**2, 1)))
    print(json.dumps(dict(completed=report['completed'], stages=summary,
                          packed_gib=round(model.bytes/2**30, 2)), indent=1))


if __name__ == '__main__':
    main()

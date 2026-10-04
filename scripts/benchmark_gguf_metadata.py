"""Compare CPU metadata loading against a saved prior reader implementation.

For the original baseline:
  git show 1e5d2c1:dllm/gguf_reader.py > /tmp/gguf-reader-before.py

This measures an empty dllm metadata cache, not a cold operating-system cache.
It does not measure weight loading, CUDA execution, or complete model activation.
"""
import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import dllm.gguf_model as module


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('--baseline-reader', required=True)
    p.add_argument('--baseline-commit', default='1e5d2c1'); p.add_argument('--output', required=True); args = p.parse_args()
    spec = importlib.util.spec_from_file_location('baseline', args.baseline_reader)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    optimized = module.BoundedReader; rows = []; expected = None
    try:
        for round in range(4):
            for name in (('baseline', 'optimized') if round % 2 == 0 else ('optimized', 'baseline')):
                module.BoundedReader = baseline.BoundedReader if name == 'baseline' else optimized
                module._plan.cache_clear()
                start = time.perf_counter(); current = module.plan(args.gguf); elapsed = time.perf_counter()-start
                if expected is None: expected = current
                assert current == expected, 'metadata, tensor descriptors or serialized tokenizer changed'
                rows.append(dict(implementation=name, seconds=elapsed)); print(json.dumps(rows[-1]), flush=True)
    finally:
        module.BoundedReader = optimized; module._plan.cache_clear()
    report = dict(passed=True, platform=platform.platform(), python=platform.python_version(),
        scope='CPU metadata/tokenizer plan; empty metadata cache, filesystem may be warm; no weight loading or GPU inference',
        baseline_commit=args.baseline_commit, exact_plans=True, trials=rows,
        medians={n:statistics.median(x['seconds'] for x in rows if x['implementation'] == n) for n in ('baseline', 'optimized')})
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()

"""Real cold-cache compiler failure: auto falls back, explicit Triton fails."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--thinking', choices=('auto', 'off'), default='auto')
    args = parser.parse_args()
    results = []
    for backend in ('auto', 'triton'):
        with tempfile.TemporaryDirectory(prefix='dllm-no-compiler-') as cache:
            env = {**os.environ, 'DLLM_KERNELS': backend, 'TRITON_CACHE_DIR': cache,
                   'CC': str(Path(cache) / 'missing-compiler')}
            result = subprocess.run([sys.executable, '-m', 'dllm', 'run', args.path,
                '--prompt', 'What is 2 + 2? Answer with one number.', '--max-tokens', '8',
                '--context', '128', '--thinking', args.thinking], capture_output=True, text=True, env=env, timeout=120)
            if backend == 'auto':
                assert result.returncode == 0, result.stderr
                final = json.loads(result.stderr.strip().splitlines()[-1])
                assert final['timing']['kernel_backend'] in ('torch', 'torch-hybrid'), final
                assert result.stdout.strip() == '4', result.stdout
                results.append({'requested': backend, 'actual': final['timing']['kernel_backend'], 'text': result.stdout.strip()})
            else:
                assert result.returncode != 0 and 'FileNotFoundError' in result.stderr, result.stderr
                results.append({'requested': backend, 'missing_compiler_rejected': True})
    print(json.dumps({'passed': True, 'results': results}, indent=2))


if __name__ == '__main__':
    main()

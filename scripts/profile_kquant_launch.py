"""Compare native CUDA launch layouts on unchanged packed checkpoint weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.testing

from dllm.gguf_model import GgufWeights
from dllm.native_kquant import library, linear_if_enabled
from dllm.runtime import require_gpu


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--libraries', nargs='+', required=True,
                        help='original control first, then isolated candidates')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(912)
    paths = [str(Path(path).resolve()) for path in args.libraries]
    # Keep every CDLL alive while CUDA graphs may reference its kernels.
    handles = [library(path) for path in paths]
    source = GgufWeights(args.model, 512, 'none')
    report = dict(model_sha256=digest(args.model), device=torch.cuda.get_device_name(),
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__,
                  triton=triton.__version__, libraries={path: digest(path) for path in paths},
                  scope='actual packed weights; bitwise original-output gate; isolated matrix timings',
                  cases=[], passed=False)
    try:
        for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', 'head'):
            weight = source.weights[name].cuda()
            layout = source.packed_formats[name]
            width, spec = layout
            outputs = spec[-1][1]
            bias = torch.randn(outputs, device='cuda', dtype=torch.float16)
            for rows in (1, 2, 4):
                x = torch.randn(rows, width, device='cuda', dtype=torch.float16)
                os.environ['DLLM_KQUANT_CUDA_LIBRARY'] = paths[0]
                expected = linear_if_enabled(x, weight, layout, bias)
                case = dict(name=name, width=width, outputs=outputs, rows=rows,
                            formats=spec, rounds=[])
                report['cases'].append(case)
                for path in paths + paths[::-1]:
                    os.environ['DLLM_KQUANT_CUDA_LIBRARY'] = path
                    actual = linear_if_enabled(x, weight, layout, bias)
                    item = dict(library=path, passed=bool(torch.isfinite(actual).all()
                                                        and torch.equal(actual, expected)))
                    case['rounds'].append(item)
                    if item['passed']:
                        item['ms'] = triton.testing.do_bench_cudagraph(
                            lambda: linear_if_enabled(x, weight, layout, bias), rep=100)
                print(json.dumps(case), flush=True)
            del weight
        report['passed'] = all(item['passed'] for case in report['cases'] for item in case['rounds'])
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    assert handles  # Handles remain in scope until every measurement completes.
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

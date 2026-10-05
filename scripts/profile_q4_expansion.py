"""Real-weight Q4 expansion timing with bit-exact independent coefficient gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton.testing
from dllm.gguf_model import GgufWeights
from dllm.quantization import dequantize_q4
from dllm.quantized_kernels_q4 import expand as original
from dllm.runtime import require_gpu
from experiments.q4_expand import expand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    source = GgufWeights(args.model, 512, 'none')
    with Path(args.model).open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(model_sha256=checksum, device=torch.cuda.get_device_name(),
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__,
                  triton=triton.__version__, passed=False, cases=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256((Path(__file__).parent/'experiments/q4_expand.py').read_bytes()).hexdigest(),
                  scope='expansion only; no dense cache; no serving performance claim')
    try:
        with torch.inference_mode():
            for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', 'head'):
                weight, scales = source.weights[name].cuda(), source.scales[name].cuda()
                if weight.dtype != torch.uint8:
                    raise ValueError('requires legacy Q4 projections')
                expected = dequantize_q4(weight, scales, torch.float16)
                functions = dict(production=lambda: original(weight, scales, torch.float16))
                functions.update({f'pairs_{block}': lambda block=block: expand(weight, scales, torch.float16, block)
                                  for block in (256, 512, 1024)})
                case = dict(name=name, shape=list(expected.shape), checks={}, rounds=[])
                report['cases'].append(case)
                for label, function in functions.items():
                    case['checks'][label] = bool(torch.equal(
                        function().view(torch.int16), expected.view(torch.int16)))
                if not all(case['checks'].values()):
                    raise AssertionError('expanded coefficients are not bit-exact')
                order = list(functions)
                for label in order + order[::-1]:
                    case['rounds'].append(dict(kind=label, ms=triton.testing.do_bench_cudagraph(functions[label], rep=100)))
                print(json.dumps(case), flush=True)
                del weight, scales, expected, functions
            report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

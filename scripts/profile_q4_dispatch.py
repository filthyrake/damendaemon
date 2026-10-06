"""Measure Q4 prefill dispatch on real weights, including expansion cost.

The candidate retains no dense projection cache. This is a matrix experiment;
model accuracy and real serving gates remain separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton.testing

from dllm.gguf_model import GgufWeights
from dllm.quantization import dequantize_q4
from dllm.quantized_kernels_q4 import expand, linear, _gemm
from dllm.runtime import require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument('--tiles', action='store_true',
                        help='Compare smaller fused tiles instead of dense expansion')
    experiment.add_argument('--launch-tuning', action='store_true',
                        help='Compare pipeline depth and output tiles with unchanged paired-dot arithmetic')
    parser.add_argument('--rows', type=int, nargs='+', default=[16, 32, 64, 128, 129, 256],
                        help='Matrix row counts for boundary and intermediate-shape sweeps')
    args = parser.parse_args()
    if any(rows < 1 for rows in args.rows):
        parser.error('row counts must be positive')
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(617)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = GgufWeights(args.model, 512, 'none')
    with Path(args.model).open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(model_sha256=checksum, device=torch.cuda.get_device_name(),
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__,
                  triton=triton.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'dllm/quantized_kernels_q4.py').read_bytes()).hexdigest(),
                  scope=('real Q4 projections; pipeline/output-tile experiment; no production changes' if args.launch_tuning else
                         'real Q4 projections; smaller fused tiles' if args.tiles else
                         'real Q4 projections; expansion included; no persistent dense cache'),
                  cases=[], passed=False)
    try:
        with torch.inference_mode():
            for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', 'head'):
                weight, scales = source.weights[name].cuda(), source.scales[name].cuda()
                if weight.dtype != torch.uint8:
                    raise ValueError('requires legacy Q4 projections')
                width, outputs = weight.shape[1]*2, weight.shape[0]
                oracle = dequantize_q4(weight, scales, torch.float16)
                bias = torch.randn(outputs, device='cuda', dtype=torch.float16)
                for rows in dict.fromkeys(args.rows):
                    x = torch.randn(rows, width, device='cuda', dtype=torch.float16)
                    expected = torch.nn.functional.linear(x, oracle, bias)
                    functions = dict(production=lambda: linear(x, weight, scales, bias),
                                     expanded=lambda: torch.nn.functional.linear(
                                         x, expand(weight, scales, x.dtype), bias))
                    case = dict(name=name, width=width, outputs=outputs, rows=rows,
                                expanded_bytes=oracle.numel()*oracle.element_size(), checks={}, rounds=[])
                    if args.tiles:
                        functions = dict(production=functions['production'])
                        for bm in (16, 32):
                            def tiled(bm=bm):
                                result = torch.empty_like(expected)
                                _gemm[(triton.cdiv(rows, bm), triton.cdiv(outputs, 64))](
                                    x, weight, scales, bias, result, rows, outputs, width,
                                    bm, 64, 128, True, num_warps=4, num_stages=3)
                                return result
                            functions[f'tile_{bm}'] = tiled
                    if args.launch_tuning:
                        functions = dict(production=functions['production'])
                        # Keep the K reduction order; vary resource use and
                        # pipelining. Large prompts still compare against the
                        # normal expansion-plus-matmul path, including its cost.
                        bm = 16 if rows <= 16 else 32 if rows <= 64 else 64
                        case['launches'] = {}
                        for bn, warps, stages in ((64, 4, 1), (64, 4, 2), (64, 4, 3),
                                                  (128, 4, 2), (128, 8, 2)):
                            label = f'm{bm}_n{bn}_w{warps}_s{stages}'
                            def tuned(bn=bn, warps=warps, stages=stages, label=label):
                                result = torch.empty_like(expected)
                                compiled = _gemm[(triton.cdiv(rows, bm), triton.cdiv(outputs, bn))](
                                    x, weight, scales, bias, result, rows, outputs, width,
                                    bm, bn, 128, True, num_warps=warps, num_stages=stages)
                                case['launches'][label] = dict(registers=compiled.n_regs,
                                                              spills=compiled.n_spills)
                                return result
                            functions[label] = tuned
                    report['cases'].append(case)
                    baseline = functions['production']()
                    for label, function in functions.items():
                        actual = function()
                        error = actual.float()-expected.float()
                        rms = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        case['checks'][label] = dict(relative_rms=rms, max_absolute=error.abs().max().item(),
                                                     exact_production=bool(torch.equal(actual, baseline)),
                                                     passed=bool(torch.isfinite(actual).all()) and rms < .001)
                    if not all(c['passed'] for c in case['checks'].values()):
                        raise AssertionError('projection accuracy gate failed')
                    order = list(functions)
                    for label in order + order[::-1]:
                        case['rounds'].append(dict(kind=label, ms=triton.testing.do_bench_cudagraph(
                            functions[label], rep=100)))
                    print(json.dumps(case), flush=True)
                del weight, scales, oracle, bias, x, expected, actual, error, functions
            report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

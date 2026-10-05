"""Real checkpoint matrix gates and ABCCBA-style split-K candidate timings."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import triton.testing
from dllm.awq import Matrix
from dllm.awq_model import AwqWeights
from dllm.awq_kernels import linear
from dllm.runtime import require_gpu
from experiments import awq_splitk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--gemv', action='store_true', help='test direct small-row reductions instead of split-K MMA')
    variant.add_argument('--gemm-tiles', action='store_true', help='test wider output tiles with original packed MMA')
    parser.add_argument('--rows', type=int, choices=(1, 2, 3, 4), nargs='+', default=[1, 2, 4])
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(917)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = AwqWeights(args.model, 2048, 'none')
    candidate = awq_splitk
    choices = (2, 4, 8, 16)
    if args.gemv:
        from experiments import awq_gemv
        candidate, choices = awq_gemv, (32, 64, 128)
    if args.gemm_tiles:
        from experiments import awq_tiles
        candidate, choices = awq_tiles, (64, 128, 256)
    report = dict(passed=False, cases=[], device=torch.cuda.get_device_name(),
                  scope='standalone packed projection accuracy and timing; not serving throughput',
                  gemv=args.gemv, gemm_tiles=args.gemm_tiles,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
    try:
        with torch.inference_mode():
            for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down'):
                host = source.awq_matrices[name]
                matrix = Matrix(host.weight.cuda(), host.zeros.cuda(), host.scales.cuda(), host.inputs, host.outputs)
                dense = host.dequantize().cuda().float()
                for rows in args.rows:
                    case = dict(name=name, inputs=matrix.inputs, outputs=matrix.outputs, rows=rows, checks=[], rounds=[])
                    report['cases'].append(case)
                    functions = {'production': lambda: linear(x, matrix, bias)}
                    for choice in choices:
                        functions[str(choice)] = lambda choice=choice: candidate.linear(x, matrix, bias, choice)
                    for exponent in (-10, 0, 8):
                        x = (torch.randn(rows, matrix.inputs, device='cuda')*2.**exponent).half()
                        for biased in (False, True):
                            bias = torch.randn(matrix.outputs, device='cuda', dtype=torch.float16) if biased else None
                            expected = torch.nn.functional.linear(x.float(), dense, None if bias is None else bias.float()).half()
                            for label, function in functions.items():
                                actual = function()
                                error = actual.float()-expected.float()
                                rms = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                                check = dict(kind=label, exponent=exponent, biased=biased, relative_rms=rms,
                                             max_absolute=error.abs().max().item(), passed=bool(torch.isfinite(actual).all()) and rms < .001)
                                case['checks'].append(check)
                                if not check['passed']:
                                    raise AssertionError(f'split-K accuracy failed: {name}, {rows}, {check}')
                    x = torch.randn(rows, matrix.inputs, device='cuda', dtype=torch.float16)
                    bias = torch.randn(matrix.outputs, device='cuda', dtype=torch.float16)
                    for label in list(functions)+list(functions)[::-1]:
                        case['rounds'].append(dict(kind=label, ms=triton.testing.do_bench_cudagraph(functions[label], rep=100)))
                    print(json.dumps(case), flush=True)
            report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with (Path(args.model)/'model.safetensors').open('rb') as stream:
            report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

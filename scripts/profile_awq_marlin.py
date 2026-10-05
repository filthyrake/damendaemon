"""Standalone native AWQ matrix qualification; not a model/runtime backend."""
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
from experiments.awq_marlin import Library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(946)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, cases=[], device=torch.cuda.get_device_name(),
                  scope='isolated Marlin device kernels via standalone C ABI; no vLLM runtime',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest())
    try:
        library = Library(args.library)
        source = AwqWeights(args.model, 2048, 'none')
        with torch.inference_mode():
            for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down'):
                host = source.awq_matrices[name]
                matrix = Matrix(host.weight.cuda(), host.zeros.cuda(), host.scales.cuda(), host.inputs, host.outputs)
                dense = host.dequantize().cuda().float()
                for biased in (False, True):
                    bias = torch.randn(matrix.outputs, device='cuda', dtype=torch.float16) if biased else None
                    prepared = library.prepare(matrix, bias)
                    for rows in (1, 2, 3, 4, 8, 16):
                        case = dict(name=name, rows=rows, biased=biased, checks=[], rounds=[])
                        report['cases'].append(case)
                        for exponent in (-10, 0, 8):
                            x = (torch.randn(rows, matrix.inputs, device='cuda')*2.**exponent).half()
                            expected = torch.nn.functional.linear(x.float(), dense, None if bias is None else bias.float()).half()
                            actual = library.linear(x, prepared)
                            torch.cuda.synchronize()
                            error = actual.float()-expected.float()
                            rms = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                            check = dict(exponent=exponent, relative_rms=rms, max_absolute=error.abs().max().item(),
                                         passed=bool(torch.isfinite(actual).all()) and rms < .001)
                            case['checks'].append(check)
                            if not check['passed']:
                                raise AssertionError(f'native AWQ accuracy failed: {name}, {rows}, {check}')
                        x = torch.randn(rows, matrix.inputs, device='cuda', dtype=torch.float16)
                        functions = dict(normal=lambda: linear(x, matrix, bias), marlin=lambda: library.linear(x, prepared))
                        for kind in ('normal', 'marlin', 'marlin', 'normal'):
                            case['rounds'].append(dict(kind=kind, ms=triton.testing.do_bench_cudagraph(functions[kind], rep=100)))
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

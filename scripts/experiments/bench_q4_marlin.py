"""Independent real-projection gate for experimental Q4_0 Marlin execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import signal
import time

import torch
import triton.testing

from dllm.gguf_model import GgufWeights
from dllm.quantization import dequantize_q4
from dllm.quantized_kernels_q4 import linear
from scripts.experiments.q4_marlin import Library


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1, 'select exactly one GPU'
    torch.set_num_threads(4)
    torch.manual_seed(619)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = GgufWeights(args.model, 512, 'none')
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(passed=False, device=torch.cuda.get_device_name(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__),
        model=str(Path(args.model).resolve()), model_sha256=digest(args.model),
        library=str(Path(args.library).resolve()), library_sha256=digest(args.library),
        scope='Real Q4_0 projections; preserved integer coefficients/signed FP16 scales; independent FP32 accumulation over FP16-rounded coefficients, final FP16 output, TF32 disabled. Warm CUDA graph ABBA excludes one-time repacking. Prepared weights are additional resident storage. No full-model or serving claim.',
        sources={path:digest(path) for path in ('scripts/experiments/q4_marlin.cu',
            'scripts/experiments/q4_marlin.py', 'scripts/experiments/bench_q4_marlin.py',
            'third_party/marlin/marlin_template.h', 'third_party/marlin/awq_repack.cuh')},
        matrices=[], checks=[], timings=[])
    def terminate(number, frame):
        report['error'] = f'terminated by signal {number}'
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM, terminate)
    try:
        library = Library(args.library)
        report['workspace_bytes'] = sum(t.numel()*t.element_size() for t in (library.scratch, library.locks))
        for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', 'head'):
            weight, scales = source.weights[name].cuda(), source.scales[name].cuda()
            n, k = weight.shape[0], weight.shape[1]*2
            oracle = dequantize_q4(weight, scales, torch.float16).float()
            bias = torch.randn(n, dtype=torch.float16, device='cuda')
            prepared = {}
            for biased in (False, True):
                torch.cuda.synchronize()
                started = time.perf_counter()
                prepared[biased] = library.prepare(weight, scales, bias if biased else None)
                torch.cuda.synchronize()
                report['matrices'].append(dict(name=name, outputs=n, width=k, bias=biased,
                    repack_seconds=time.perf_counter()-started,
                    canonical_bytes=weight.numel()*weight.element_size()+scales.numel()*scales.element_size(),
                    additional_prepared_bytes=prepared[biased].bytes,
                    negative_scale_count=int((scales<0).sum().item())))
            for rows in (2, 4, 16, 64, 129, 400):
                x = torch.randn(rows, k, device='cuda', dtype=torch.float16)
                for exponent in (-14, 0, 10):
                    value = (x.float()*2.**exponent).half()
                    for biased in (False, True):
                        offset = bias if biased else None
                        # FP16 GEMM itself deviates in the subnormal stress
                        # case. FP32 accumulation was checked against FP64;
                        # retain the same coefficient/final-output rounding.
                        expected = torch.nn.functional.linear(value.float(), oracle,
                            offset.float() if offset is not None else None).half()
                        baseline = linear(value, weight, scales, offset)
                        actual = library.linear(value, prepared[biased])
                        error = actual.float()-expected.float()
                        rms = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        control_rms = ((baseline.float()-expected.float()).square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        passed = bool(torch.isfinite(actual).all()) and rms<.001 and control_rms<.001
                        report['checks'].append(dict(name=name, rows=rows, exponent=exponent, bias=biased,
                            relative_rms=rms, control_relative_rms=control_rms,
                            exact_production=bool(torch.equal(actual, baseline)), passed=passed))
                        assert passed, report['checks'][-1]
                functions = dict(production=lambda:linear(x, weight, scales, bias),
                                 marlin=lambda:library.linear(x, prepared[True]))
                samples = {key:[] for key in functions}
                for mode in ('production','marlin','marlin','production'):
                    samples[mode].append(triton.testing.do_bench_cudagraph(functions[mode], rep=100))
                report['timings'].append(dict(name=name, rows=rows, samples_ms=samples,
                    production_ms=statistics.median(samples['production']),
                    marlin_ms=statistics.median(samples['marlin'])))
                print(name, rows, 'passed', flush=True)
                Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
            del weight, scales, oracle, bias, prepared, x, value, expected, actual, baseline, functions
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

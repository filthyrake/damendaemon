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
    parser.add_argument('--rows', type=int, nargs='+', default=[2, 4, 16, 64, 129, 400])
    parser.add_argument('--residual-baseline', help='existing native residual library for single-row controls only')
    parser.add_argument('--production',action='store_true',help='qualify the normal unbiased production adapter and its exact prepared buffers')
    parser.add_argument('--direct-repack', action='store_true', help='also require exact prepared weights/scales/zeros/bias against the original adapter')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.production and args.direct_repack:parser.error('--production excludes --direct-repack')
    if any(row < 1 or row > 512 for row in args.rows):
        parser.error('rows must be between 1 and 512')
    if args.residual_baseline and 1 not in args.rows:
        parser.error('--residual-baseline requires a single-row case')
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
    report['residual_baseline'] = (dict(path=str(Path(args.residual_baseline).resolve()),
        sha256=digest(args.residual_baseline), arithmetic='existing two-term residual, FP32 coefficient scaling; row 1 only')
        if args.residual_baseline else None)
    def terminate(number, frame):
        report['error'] = f'terminated by signal {number}'
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM, terminate)
    try:
        reference_library = Library(args.library) if args.direct_repack or args.production else None
        biased_cases=(False,) if args.production else (False,True)
        if args.production:
            from dllm.native_q4_marlin import Library as ProductionLibrary
            library=ProductionLibrary(args.library)
            library.allocate_workspace(torch.device('cuda',torch.cuda.current_device()))
            report['production_sources']={name:digest(name) for name in
                ('dllm/native_q4_marlin.py','dllm/q4_marlin_repack.py','native/q4_marlin.cu')}
        elif args.direct_repack:
            from scripts.experiments.q4_marlin_direct import DirectLibrary
            library = DirectLibrary(args.library)
            report['direct_repack_sources'] = {name:digest(name) for name in
                ('scripts/experiments/q4_marlin_direct.py','scripts/experiments/q4_marlin_repack.py')}
        else:
            library = Library(args.library)
        report['workspace_bytes'] = sum(t.numel()*t.element_size() for t in (library.scratch, library.locks))
        for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', 'head'):
            weight, scales = source.weights[name].cuda(), source.scales[name].cuda()
            n, k = weight.shape[0], weight.shape[1]*2
            oracle = dequantize_q4(weight, scales, torch.float16).float()
            def control(value, offset):
                if value.shape[0] == 1 and args.residual_baseline:
                    from dllm.native_q4_residual import linear as residual_linear
                    return residual_linear(value, weight, scales, offset, library_path=args.residual_baseline)
                return linear(value, weight, scales, offset)
            bias = torch.randn(n, dtype=torch.float16, device='cuda')
            prepared = {}
            for biased in biased_cases:
                torch.cuda.synchronize()
                started = time.perf_counter()
                prepared[biased] = (library.prepare(weight,scales) if args.production else
                                    library.prepare(weight, scales, bias if biased else None))
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                exact_preparation = None
                if reference_library is not None:
                    reference_prepared = reference_library.prepare(weight,scales,bias if biased else None)
                    exact_preparation = all(torch.equal(getattr(prepared[biased],key),getattr(reference_prepared,key))
                        for key in ('weight','scales','zeros',*(['bias'] if biased else [])))
                    assert exact_preparation, (name,biased,'prepared buffers changed')
                    del reference_prepared
                report['matrices'].append(dict(name=name, outputs=n, width=k, bias=biased,
                    repack_seconds=elapsed, exact_original_preparation=exact_preparation,
                    canonical_bytes=weight.numel()*weight.element_size()+scales.numel()*scales.element_size(),
                    additional_prepared_bytes=prepared[biased].bytes,
                    negative_scale_count=int((scales<0).sum().item())))
            functions = None
            for rows in dict.fromkeys(args.rows):
                x = torch.randn(rows, k, device='cuda', dtype=torch.float16)
                row_candidate_passed = True
                timing_scale_control_passed = True
                for exponent in (-14, 0, 10):
                    value = (x.float()*2.**exponent).half()
                    for biased in biased_cases:
                        offset = bias if biased else None
                        # FP16 GEMM itself deviates in the subnormal stress
                        # case. FP32 accumulation was checked against FP64;
                        # retain the same coefficient/final-output rounding.
                        expected = torch.nn.functional.linear(value.float(), oracle,
                            offset.float() if offset is not None else None).half()
                        baseline = control(value, offset)
                        actual = library.linear(value, prepared[biased])
                        error = actual.float()-expected.float()
                        rms = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        control_rms = ((baseline.float()-expected.float()).square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        candidate_passed = bool(torch.isfinite(actual).all()) and rms<.001
                        control_passed = bool(torch.isfinite(baseline).all()) and control_rms<.001
                        row_candidate_passed &= candidate_passed
                        if exponent == 0:
                            timing_scale_control_passed &= control_passed
                        report['checks'].append(dict(name=name, rows=rows, exponent=exponent, bias=biased,
                            relative_rms=rms, control_relative_rms=control_rms,
                            exact_production=bool(torch.equal(actual, baseline)),
                            candidate_passed=candidate_passed, control_passed=control_passed,
                            passed=candidate_passed and control_passed))
                if not row_candidate_passed or not timing_scale_control_passed:
                    print(name, rows, 'accuracy failure; timing skipped', flush=True)
                    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
                    continue
                functions = dict(production=lambda:control(x, None if args.production else bias),
                                 marlin=lambda:library.linear(x, prepared[not args.production]))
                samples = {key:[] for key in functions}
                for mode in ('production','marlin','marlin','production'):
                    samples[mode].append(triton.testing.do_bench_cudagraph(functions[mode], rep=100))
                report['timings'].append(dict(name=name, rows=rows, samples_ms=samples,
                    production_ms=statistics.median(samples['production']),
                    marlin_ms=statistics.median(samples['marlin'])))
                print(name, rows, 'candidate passed; all control failures retained', flush=True)
                Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
            del weight, scales, oracle, bias, prepared, x, value, expected, actual, baseline, functions
        report['candidate_passed'] = all(c['candidate_passed'] for c in report['checks'])
        report['control_passed'] = all(c['control_passed'] for c in report['checks'])
        report['passed'] = report['candidate_passed'] and report['control_passed']
        if not report['passed']:
            raise AssertionError('complete matrix sweep has accuracy failures; inspect candidate/control fields')
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Qualify standalone AWQ GPU kernels; no model-support or engine-speed claim."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import triton.testing
from safetensors import safe_open
from dllm.awq import Matrix, fuse, group_size
from dllm.awq_kernels import expand, linear
from dllm.runtime import require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--upstream-source', required=True, help='pinned vLLM awq_triton.py, validation only')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(839)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(args.model)
    config = json.loads((path/'config.json').read_text())
    group = group_size(config['quantization_config'])
    reference_source = Path(args.upstream_source).read_text()
    dependency = 'from vllm.triton_utils import tl, triton'
    if reference_source.count(dependency) != 1:
        raise ValueError('unexpected upstream validation source')
    adapted = reference_source.replace(dependency, 'import triton\nimport triton.language as tl')
    report = dict(passed=False, cases=[], coefficient_checks=[],
                  device=torch.cuda.get_device_name(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, triton=triton.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'dllm/awq_kernels.py').read_bytes()).hexdigest(),
                  upstream_source_sha256=hashlib.sha256(reference_source.encode()).hexdigest(),
                  adapted_source_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                  upstream_adaptation='replace only the Triton import; preserve the implementation and copyright header',
                  scope='first-layer real/fused projections; FP16 coefficients, FP32 accumulation; no model or serving qualification')
    with (path/'model.safetensors').open('rb') as stream:
        report['model_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-awq-reference-') as temporary:
            reference_path = Path(temporary)/'awq_reference.py'
            reference_path.write_text(adapted)
            spec = importlib.util.spec_from_file_location('dllm_awq_validation_reference', reference_path)
            oracle_module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = oracle_module
            spec.loader.exec_module(oracle_module)
            with safe_open(str(path/'model.safetensors'), framework='pt', device='cpu') as source:
                matrices = {}
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
                    parent = 'self_attn' if name in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'
                    prefix = f'model.layers.0.{parent}.{name}'
                    weight, zeros, scales = (source.get_tensor(prefix+suffix) for suffix in ('.qweight', '.qzeros', '.scales'))
                    matrices[name] = Matrix(weight, zeros, scales, weight.shape[0], weight.shape[1]*8, group)
                matrices['qkv'] = fuse(matrices[name] for name in ('q_proj', 'k_proj', 'v_proj'))
                matrices['gate_up'] = fuse(matrices[name] for name in ('gate_proj', 'up_proj'))
            with torch.inference_mode():
                for name, host in matrices.items():
                    matrix = Matrix(host.weight.cuda(), host.zeros.cuda(), host.scales.cuda(), host.inputs, host.outputs, host.group)
                    expected_weight = host.dequantize().cuda()
                    dllm_weight = expand(matrix)
                    upstream_weight = oracle_module.awq_dequantize_triton(matrix.weight, matrix.scales, matrix.zeros).T
                    checks = dict(name=name,
                                  dllm_exact=torch.equal(dllm_weight.contiguous().view(torch.int16), expected_weight.view(torch.int16)),
                                  upstream_exact=torch.equal(upstream_weight.contiguous().view(torch.int16), expected_weight.view(torch.int16)))
                    report['coefficient_checks'].append(checks)
                    if not checks['dllm_exact'] or not checks['upstream_exact']:
                        raise AssertionError(f'coefficient bit-pattern mismatch: {checks}')
                    for rows in (1, 2, 4, 33, 64, 129, 256):
                        case = dict(name=name, inputs=host.inputs, outputs=host.outputs, rows=rows, checks=[], rounds=[])
                        report['cases'].append(case)
                        for exponent in (-10, 0, 8):
                            x = (torch.randn(rows, host.inputs, device='cuda') * 2.**exponent).half()
                            for biased in (False, True):
                                bias = torch.randn(host.outputs, device='cuda', dtype=torch.float16) if biased else None
                                expected = torch.nn.functional.linear(
                                    x.float(), expected_weight.float(), None if bias is None else bias.float()).half()
                                actual = linear(x, matrix, bias)
                                error = actual.float() - expected.float()
                                rms = (error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                                check = dict(exponent=exponent, biased=biased, relative_rms=rms,
                                             max_absolute=error.abs().max().item(), passed=bool(torch.isfinite(actual).all()) and rms < .001)
                                case['checks'].append(check)
                                if not check['passed']:
                                    raise AssertionError(f'AWQ matrix accuracy failed: {name}, rows={rows}, {check}')
                        # Time ordinary-magnitude inputs separately from the stress cases.
                        x = torch.randn(rows, host.inputs, device='cuda', dtype=torch.float16)
                        bias = torch.randn(host.outputs, device='cuda', dtype=torch.float16)
                        functions = dict(packed=lambda: linear(x, matrix, bias),
                                         expanded=lambda: torch.nn.functional.linear(x, expand(matrix), bias))
                        for kind in ('packed', 'expanded', 'expanded', 'packed'):
                            case['rounds'].append(dict(kind=kind, ms=triton.testing.do_bench_cudagraph(functions[kind], rep=100)))
                        print(json.dumps(case), flush=True)
                    del matrix, expected_weight, dllm_weight, upstream_weight, expected, actual, error, functions
            report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

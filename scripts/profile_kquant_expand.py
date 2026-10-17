"""Measure expansion tile layouts on unchanged checkpoint coefficients.

This is an isolated matrix experiment. Exact agreement with current expansion
is required before timing; it does not establish model or HTTP performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton as tr
import triton.testing

from damendaemon import kquant_kernels
from damendaemon.gguf_model import GgufWeights
from damendaemon.kquant_kernels import _expand, expand
from damendaemon.runtime import require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    source = GgufWeights(args.model, 512, 'none')
    with open(args.model, 'rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    report = dict(model_sha256=digest, device=torch.cuda.get_device_name(),
                  gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  kernel_sha256=hashlib.sha256(Path(kquant_kernels.__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, triton=tr.__version__, cases=[],
                  scope='FP16 expansion only; exact current-coefficient equality; no dispatch change')
    try:
        for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', 'head'):
            weight = source.weights[name].cuda()
            layout = source.packed_formats[name]
            width, spec = layout
            outputs = spec[-1][1]
            expected = expand(weight, layout, torch.float16)
            actual = torch.empty_like(expected)
            case = dict(name=name, width=width, outputs=outputs, formats=spec,
                        candidates=[], baseline_ms=triton.testing.do_bench_cudagraph(
                            lambda: expand(weight, layout, torch.float16), rep=100))
            report['cases'].append(case)
            for br in (4, 8, 16):
                for bc in (256, 512):
                    for warps in (4, 8):
                        item = dict(br=br, bc=bc, warps=warps)
                        case['candidates'].append(item)
                        try:
                            def launch():
                                return _expand[(tr.cdiv(outputs, br), tr.cdiv(width, bc))](
                                    weight, actual, outputs, width, spec, br, bc,
                                    num_warps=warps, enable_fp_fusion=False)
                            compiled = launch()
                            item.update(passed=bool(torch.equal(actual, expected)),
                                        registers=compiled.n_regs, spills=compiled.n_spills)
                            if item['passed']:
                                item['ms'] = triton.testing.do_bench_cudagraph(launch, rep=100)
                        except Exception as error:
                            item.update(passed=False, error=repr(error))
            good = [item for item in case['candidates'] if item['passed']]
            print(json.dumps(dict(name=name, baseline_ms=case['baseline_ms'],
                                  best=min(good, key=lambda item: item['ms']) if good else None)), flush=True)
            del weight, expected, actual
        report['passed'] = all(item['passed'] for case in report['cases'] for item in case['candidates'])
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

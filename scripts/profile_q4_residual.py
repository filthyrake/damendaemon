"""Measure an isolated Q4 integer-dot candidate against production Q4 kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton.testing
from dllm.runtime import require_gpu
from dllm.quantization import quantize_q4, dequantize_q4
from dllm.quantized_kernels_q4 import linear as production
from experiments.q4_residual import linear as candidate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    require_gpu();torch.set_num_threads(4);torch.manual_seed(713)
    torch.backends.cuda.matmul.allow_tf32=False
    path=Path(os.environ['DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY'])
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,passed=False,
                candidate_library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_library_sha256=hashlib.sha256(Path(os.environ['DLLM_Q4_CUDA_LIBRARY']).read_bytes()).hexdigest(),
                scope='synthetic matrices; FP16-rounded coefficient oracle; not model or serving qualification',checks=[])
    try:
        for n,k in ((113,4096),(6144,4096),(4096,4096),(28672,4096),(4096,14336),(32000,4096)):
            weights,scales=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
            decoded=dequantize_q4(weights,scales,torch.float16)
            for rows in (1,2,4):
                x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
                bias=torch.randn(n,device='cuda',dtype=torch.float16)
                for exponent in (-14,0,10):
                    value=(x.float()*(2.**exponent)).half()
                    for with_bias in (False,True):
                        b=bias if with_bias else None
                        expected=torch.nn.functional.linear(value,decoded,b)
                        actual=candidate(value,weights,scales,b)
                        error=actual.float()-expected.float()
                        rms=(error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        item=dict(outputs=n,width=k,rows=rows,activation_exponent=exponent,bias=with_bias,
                                  max_absolute=error.abs().max().item(),relative_rms=rms,
                                  finite=bool(torch.isfinite(actual).all()),passed=bool(torch.isfinite(actual).all()) and rms<.001)
                        report['checks'].append(item)
                timing=dict(outputs=n,width=k,rows=rows,
                    production_ms=triton.testing.do_bench_cudagraph(lambda:production(x,weights,scales,bias),rep=100),
                    candidate_ms=triton.testing.do_bench_cudagraph(lambda:candidate(x,weights,scales,bias),rep=100))
                report.setdefault('timings',[]).append(timing);print(json.dumps(timing),flush=True)
            del weights,scales,decoded
        report['passed']=all(c['passed'] for c in report['checks'])
        if not report['passed']:raise AssertionError('candidate matrix gate failed')
    except Exception as error:
        report['error']=repr(error);raise
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

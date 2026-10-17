"""Bounded real-HIP correctness/allocation probe, not an inference benchmark.

Run in an isolated ROCm PyTorch environment with one explicitly selected device.
No package installation, driver configuration or service changes are performed.
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform

import torch
from torch.nn import functional as F


def check(name, actual, expected):
    actual=actual.detach().cpu().double()
    expected=expected.double()
    delta=actual-expected
    rms=(delta.square().mean()/expected.square().mean().clamp_min(1e-12)).sqrt().item()
    maximum=delta.abs().max().item()
    return dict(name=name,relative_rms=rms,max_absolute=maximum,
                passed=bool(torch.isfinite(actual).all()) and rms<=0.002 and maximum<=0.05)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--release-blas-workspace',action='store_true',help='diagnostic cleanup after all probe tensors are released; no runtime policy change')
    args=parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError('probe requires a ROCm PyTorch build')
    if os.environ.get('ROCR_VISIBLE_DEVICES')!='0' or os.environ.get('HIP_VISIBLE_DEVICES')!='0':
        raise ValueError('explicitly select physical device zero through ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('exactly one HIP GPU must be visible')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    report=dict(passed=False,scope='HIP primitive correctness and release only; no damendaemon model or performance qualification',
        python=platform.python_version(),torch=torch.__version__,hip=torch.version.hip,
        device=torch.cuda.get_device_name(0),architecture=getattr(props,'gcnArchName',None),
        reported_device_memory=props.total_memory,mem_get_info=list(torch.cuda.mem_get_info()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        release_blas_workspace=args.release_blas_workspace,checks=[],cycles=[])
    rng=torch.Generator(device='cpu').manual_seed(731)
    try:
        with torch.inference_mode():
            for cycle in range(2):
                torch.cuda.empty_cache()
                before=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                a=torch.randn((128,512),generator=rng).half()/16
                b=torch.randn((512,256),generator=rng).half()/16
                x,y=a.cuda(),b.cuda()
                output=x@y
                report['checks'].append(check(f'matmul-{cycle}',output,(a.double()@b.double()).half()))
                q=torch.randn((1,8,64,64),generator=rng).half()
                k=torch.randn((1,2,64,64),generator=rng).half()
                v=torch.randn((1,2,64,64),generator=rng).half()
                qg,kg,vg=q.cuda(),k.cuda(),v.cuda()
                attention=F.scaled_dot_product_attention(qg,kg,vg,is_causal=True,enable_gqa=True)
                scores=q.double()@k.double().repeat_interleave(4,dim=1).transpose(-1,-2)/8
                mask=torch.ones(64,64,dtype=torch.bool).triu(1)
                scores.masked_fill_(mask,float('-inf'))
                expected=(scores.softmax(-1)@v.double().repeat_interleave(4,dim=1)).half()
                report['checks'].append(check(f'grouped-causal-attention-{cycle}',attention,expected))
                normalized=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).half()
                expected_norm=(a.double()*torch.rsqrt(a.double().square().mean(-1,keepdim=True)+1e-6)).half()
                report['checks'].append(check(f'rmsnorm-{cycle}',normalized,expected_norm))
                # Modest real allocation exercises allocator accounting; do not
                # infer the full APU shared-memory limit from this successful size.
                reserve=torch.empty(64*1024**2,dtype=torch.uint8,device='cuda')
                reserve.fill_(37)
                assert reserve[-1].item()==37
                torch.cuda.synchronize()
                allocated=torch.cuda.max_memory_allocated()
                reserved=torch.cuda.max_memory_reserved()
                del x,y,output,qg,kg,vg,attention,normalized,reserve
                gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
                before_workspace_release=torch.cuda.memory_allocated()
                if args.release_blas_workspace:
                    torch._C._cuda_clearCublasWorkspaces()
                    torch.cuda.synchronize();torch.cuda.empty_cache()
                after=torch.cuda.memory_allocated()
                report['cycles'].append(dict(before_allocated=before,peak_allocated=allocated,
                    peak_reserved=reserved,before_workspace_release=before_workspace_release,
                    after_allocated=after,released=after==before))
                if allocated-before>2*1024**3:raise AssertionError('probe exceeded 2 GiB allocation bound')
            report['passed']=all(c['passed'] for c in report['checks']) and all(c['released'] for c in report['cycles'])
            if not report['passed']:raise AssertionError('HIP correctness or release gate failed')
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

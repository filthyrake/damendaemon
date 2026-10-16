"""Real GPU dispatch-shape experiment, separate from model/server performance."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import signal
import tempfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--large',action='store_true');args=p.parse_args()
    def terminate(number,frame):raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,terminate)
    # Own only this disposable compiler cache, never invalidate the user's cache.
    with tempfile.TemporaryDirectory(prefix='dllm-q4-shape-') as cache:
        os.environ['TRITON_CACHE_DIR']=cache
        import torch
        import triton as tr
        import triton.testing
        from dllm.q4_stage import repack_adjacent
        from dllm.q4_stage_kernels import single_dot_adjacent
        from dllm.quantization import quantize_q4,dequantize_q4
        from scripts.experiments.q4_dynamic_rows import single_dot_dynamic_rows
        assert torch.cuda.device_count()==1,'select exactly one GPU'
        torch.set_num_threads(4);torch.manual_seed(43)
        report=dict(passed=False,device=torch.cuda.get_device_name(),checks=[],
            torch=str(torch.__version__),triton=tr.__version__,large=args.large,
            scope='Real CUDA matrix experiment. Fresh private compiler cache; The first row shape of each projection primes both kernels outside timing. First dispatch includes host specialization/loading and GPU completion. Warm timings use CUDA graphs. Not model throughput or complete cold startup.',
            sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                     ('dllm/q4_stage_kernels.py','scripts/experiments/q4_dynamic_rows.py',__file__)})
        try:
            with torch.inference_mode():
                for n,k in ((1152,896),(896,896),(9728,896),(896,4864)):
                    canonical,scales=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
                    dense=dequantize_q4(canonical,scales,torch.float16);weight=repack_adjacent(canonical)
                    bias=torch.randn(n,device='cuda',dtype=torch.float16)*.01 if n==1152 else None
                    for index,rows in enumerate((32,33,47,63,64,65,95,127,128) if args.large else (2,1,3,4,7,16,17,19,32)):
                        x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
                        target=torch.nn.functional.linear(x,dense,bias)
                        outputs={name:torch.empty_like(target) for name in ('static','dynamic')}
                        bm=max(16,tr.next_power_of_2(rows))
                        bn,bk,stages=(64,64,3) if k==896 else ((32,128,1) if rows<=16 else (64,128,3))
                        kernels={'static':single_dot_adjacent,'dynamic':single_dot_dynamic_rows}
                        launches={};case=dict(outputs=n,width=k,rows=rows,priming=index==0,kernels={})
                        for name in (('static','dynamic') if index%2==0 else ('dynamic','static')):
                            def launch(name=name):
                                return kernels[name][(tr.cdiv(rows,bm),tr.cdiv(n,bn))](x,weight,scales,
                                    weight if bias is None else bias,outputs[name],rows,n,k,bm,bn,bk,bias is not None,
                                    num_warps=8 if bm==128 else 4,num_stages=stages)
                            torch.cuda.synchronize();started=time.perf_counter()
                            compiled=launch();torch.cuda.synchronize()
                            wall=1000*(time.perf_counter()-started)
                            error=((outputs[name].float()-target.float()).square().mean()/target.float().square().mean()).sqrt().item()
                            case['kernels'][name]=dict(first_dispatch_ms=wall,relative_rms=error,
                                assembly_sha256=hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest(),
                                registers=compiled.n_regs,spills=compiled.n_spills,warm_ms=[])
                            assert error<.001,(n,k,rows,name,error)
                            launches[name]=launch
                        case['exact_static_output']=bool(torch.equal(outputs['static'],outputs['dynamic']))
                        assert case['exact_static_output'],(n,k,rows)
                        for name in ('static','dynamic','dynamic','static'):
                            case['kernels'][name]['warm_ms'].append(tr.testing.do_bench_cudagraph(launches[name],rep=100))
                        for result in case['kernels'].values():result['median_warm_ms']=statistics.median(result['warm_ms'])
                        report['checks'].append(case)
                        print(n,k,rows,'passed',flush=True)
                report['passed']=True
        finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

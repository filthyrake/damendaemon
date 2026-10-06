"""Real two-GPU rejection proof for mismatched resident session bounds.

Each mismatch owns a bounded torchrun job. Both ranks must reject before the
command channel opens, then exit successfully from the expected failure path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',required=True)
    p.add_argument('--worker',choices=('capacity','context','prefill_policy','q4_layout'))
    args=p.parse_args();output=Path(args.output)
    if args.worker:
        from dllm.distributed import StagePair
        rank=int(os.environ['RANK'])
        capacity=1 if args.worker=='capacity' and rank==1 else 4
        context=256 if args.worker=='context' and rank==1 else 512
        if args.worker=='prefill_policy':os.environ['DLLM_PARTITION_PACKED_PREFILL']=str(1-rank)
        if args.worker=='q4_layout':os.environ['DLLM_Q4_STAGE_ADJACENT']=str(1-rank)
        quantization='q4_0' if args.worker=='q4_layout' else 'q8_0'
        try:
            pair=StagePair(args.model,context,quantization,lambda model:[],
                           session='bounds-'+args.worker,max_batch=capacity)
        except ValueError as error:
            if str(error)!='distributed model or session bounds mismatch':raise
            report=dict(passed=True,rank=rank,mismatch=args.worker,capacity=capacity,context=context,
                        rejection=str(error),device=os.environ['CUDA_VISIBLE_DEVICES'])
            output.with_name(output.stem+f'-{args.worker}-rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
            return
        pair.close()
        raise AssertionError('mismatched session was accepted')
    from distributed_job import run_job
    report=dict(passed=False,scope='Real GPU startup rejection and cleanup, not performance evidence',
                source_sha256=hashlib.sha256(Path('dllm/distributed.py').read_bytes()).hexdigest(),checks=[])
    try:
        for mismatch in ('capacity','context','prefill_policy','q4_layout'):
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node','2',
                '--run-path',str(Path(__file__).resolve()),args.model,'--output',str(output),'--worker',mismatch]
            try:result=run_job(command,timeout=180)
            except subprocess.CalledProcessError as error:
                print(error.stdout,file=sys.stderr);print(error.stderr,file=sys.stderr);raise
            for rank in (0,1):
                item=json.loads(output.with_name(output.stem+f'-{mismatch}-rank{rank}.json').read_text())
                assert item['passed'];report['checks'].append(item)
            print(mismatch,'both ranks rejected',flush=True)
        report['passed']=True
    finally:output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

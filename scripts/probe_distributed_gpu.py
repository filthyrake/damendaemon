"""Bounded two-worker CUDA/NCCL correctness and communication measurements.

Launch one process on each explicitly selected GPU. Socket and IB are forced
transport choices. The NCCL debug transcript is captured verbatim per rank
and classified by damendaemon.nccl_transport_evidence: the selected network
transport is recorded from NCCL's own selection lines, and GPUDirect RDMA is
recorded as evidenced ONLY by NCCL's per-HCA capability lines ('evidenced'
still means NCCL's own check, not a per-transfer DMA-bypass confirmation).
RDMA success alone does not establish GPUDirect RDMA; RDMA stays non-default
and unqualified until a real RDMA pair runs it (#602, inventory-gated).
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import faulthandler
import gc
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import time
import torch
import torch.distributed as dist
from damendaemon.nccl_transport_evidence import (StderrTranscript,
                                                 parse_transport_evidence,
                                                 rdma_devices)


def main():
    faulthandler.enable()
    p=argparse.ArgumentParser()
    p.add_argument('--rank',required=True,type=int,choices=(0,1))
    p.add_argument('--rendezvous',required=True,help='tcp://coordinator-private-address:port')
    p.add_argument('--transport',choices=('auto','socket','rdma'),default='auto')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not args.rendezvous.startswith('tcp://'):p.error('explicit TCP bootstrap address required')
    if torch.cuda.device_count()!=1:p.error('select exactly one GPU per worker with CUDA_VISIBLE_DEVICES')
    devices=rdma_devices()
    if args.transport=='rdma' and not devices:
        p.error('RDMA requested but no RDMA device is exposed to this worker')
    if args.transport!='auto':os.environ['NCCL_NET']='Socket' if args.transport=='socket' else 'IB'
    os.environ.setdefault('NCCL_DEBUG','INFO')
    os.environ.setdefault('NCCL_DEBUG_SUBSYS','INIT,NET')
    torch.cuda.set_device(0)
    transcript=StderrTranscript(args.output+'.nccl-debug.log')
    transcript.install()
    report=dict(passed=False,host=socket.gethostname(),rank=args.rank,world_size=2,
        backend='nccl',requested_transport=args.transport,rdma_devices=devices,
        gpudirect_rdma='pending_transcript',
        torch=str(torch.__version__),nccl=torch.cuda.nccl.version(),
        device=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        measurements=[],scope='GPU collectives and point-to-point probe; not model inference')
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    try:
        free,total=torch.cuda.mem_get_info()
        report.update(free_bytes=free,total_bytes=total)
        if free<512*1024**2:raise MemoryError('at least 512 MiB spare memory required')
        dist.init_process_group('nccl',init_method=args.rendezvous,rank=args.rank,world_size=2,
            timeout=timedelta(seconds=30),device_id=torch.device('cuda:0'))
        for size in (4096,65536,1048576,4194304):
            source=torch.empty(size//4,device='cuda',dtype=torch.float32)
            destination=torch.empty_like(source)
            for operation in ('all_reduce','round_trip'):
                print(f'rank={args.rank} operation={operation} bytes={size}',flush=True)
                times=[]
                for iteration in range(23):
                    source.fill_(args.rank+1);destination.zero_()
                    torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
                    start=time.perf_counter()
                    if operation=='all_reduce':dist.all_reduce(source)
                    else:
                        outward = dist.P2POp(dist.isend, source, 1) if args.rank == 0 else dist.P2POp(dist.irecv, destination, 0)
                        for work in dist.batch_isend_irecv([outward]): work.wait()
                        returning = dist.P2POp(dist.irecv, destination, 1) if args.rank == 0 else dist.P2POp(dist.isend, destination, 0)
                        for work in dist.batch_isend_irecv([returning]): work.wait()
                        del outward, returning, work
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    actual=source if operation=='all_reduce' else destination
                    expected=3 if operation=='all_reduce' else 1
                    assert bool((actual==expected).all()),'distributed tensor mismatch'
                    if iteration>=3:times.append(elapsed)
                median=statistics.median(times)
                report['measurements'].append(dict(operation=operation,bytes=size,seconds=times,
                    median_seconds=median,
                    timing_scope=('coordinator_round_trip' if args.rank == 0 else 'responder_participation')
                        if operation == 'round_trip' else 'local_collective_completion'))
                Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
            del source,destination,actual
        dist.barrier();dist.destroy_process_group();gc.collect();torch.cuda.synchronize()
        report['final_tensor_allocated_bytes']=torch.cuda.memory_allocated()
        report['passed']=True
    except BaseException as error:report['error']=repr(error);raise
    finally:
        stalled=False
        try:stalled=transcript.close()
        finally:
            report['transcript_stall']=bool(stalled)
            lines=Path(args.output+'.nccl-debug.log').read_text(errors='replace').splitlines()
            report['transport_evidence']=parse_transport_evidence(lines)
            report['gpudirect_rdma']=report['transport_evidence']['gpudirect_rdma']
            import hashlib
            report['transcript_sha256']=hashlib.sha256(
                Path(args.output+'.nccl-debug.log').read_bytes()).hexdigest()
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()

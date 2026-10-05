"""Bounded two-GPU copy latency/throughput and peer-access probe."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args()
    if torch.cuda.device_count()!=2:raise ValueError('select exactly two GPUs through CUDA_VISIBLE_DEVICES')
    devices=[torch.device('cuda',i) for i in range(2)]
    def sync():
        for device in devices:torch.cuda.synchronize(device)
    report=dict(passed=False,torch=str(torch.__version__),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        devices=[],copies=[],scope='bounded torch cross-device copies; not model or collective performance')
    for device in devices:
        with torch.cuda.device(device):
            free,total=torch.cuda.mem_get_info()
            report['devices'].append(dict(index=device.index,name=torch.cuda.get_device_name(device),free_bytes=free,total_bytes=total))
            if free<512*1024**2:raise MemoryError('probe requires at least 512 MiB spare memory on each device')
    try:
        for source,destination in ((0,1),(1,0)):
            peer=torch.cuda.can_device_access_peer(source,destination)
            for size in (4096,65536,1048576,4194304):
                src=torch.full((size//4,),source+1,dtype=torch.float32,device=devices[source])
                dst=torch.zeros_like(src,device=devices[destination])
                sync()
                for _ in range(3):dst.copy_(src,non_blocking=True)
                sync();times=[]
                for _ in range(20):
                    start=time.perf_counter();dst.copy_(src,non_blocking=True);sync()
                    times.append(time.perf_counter()-start)
                assert torch.equal(dst.cpu(),src.cpu()),'cross-device copy mismatch'
                median=statistics.median(times)
                report['copies'].append(dict(source=source,destination=destination,peer_access=peer,
                    bytes=size,seconds=times,median_seconds=median,median_gbytes_per_second=size/median/1e9))
                del src,dst
        sync()
        report['final_allocated_bytes']=[torch.cuda.memory_allocated(device) for device in devices]
        assert report['final_allocated_bytes']==[0,0]
        report['passed']=True
    except BaseException as error:report['error']=repr(error);raise
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

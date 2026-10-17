"""Independent FP64 checks and matched A2 timings for a sink-attention experiment."""
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
from damendaemon.kernels import attention as original
from scripts.experiments.sink_attention import attention as fused,bounded_attention


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bounded',action='store_true');p.add_argument('--production',action='store_true');args=p.parse_args()
    implementation=bounded_attention if args.bounded else fused
    if args.production:
        from damendaemon.sink_attention import prefill_attention
        implementation=prefill_attention
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.manual_seed(43);rows=[]
    with torch.inference_mode():
        for count in (129,513,4096):
            context=1<<(count-1).bit_length()
            q=torch.randn(count,64,1,64,device='cuda')
            k=torch.randn(1,8,context,64,device='cuda');v=torch.randn_like(k)
            positions=torch.arange(count,device='cuda');slots=torch.zeros_like(positions)
            sinks=torch.linspace(-100,100,64,device='cuda')
            for window in (0,128):
                def baseline():return original(q,k,v,positions,slots,window=window,sinks=sinks,scratch_bytes=32*1024**2)
                def candidate():return implementation(q,k[0,:,:count],v[0,:,:count],sinks,window)
                expected=baseline();actual=candidate()
                torch.testing.assert_close(actual,expected,rtol=4e-5,atol=2e-6)
                errors=[]
                for index in (0,min(127,count-1),count-1):
                    query=q[index,:,0].cpu().double();key=k[0,:,:count].cpu().double().repeat_interleave(8,dim=0)
                    value=v[0,:,:count].cpu().double().repeat_interleave(8,dim=0)
                    score=torch.einsum('hd,htd->ht',query,key)*64**-.5
                    valid=torch.arange(count)<=index
                    if window:valid &= torch.arange(count)>index-window
                    score.masked_fill_(~valid[None,:],-float('inf'))
                    probability=torch.cat((score,sinks.cpu().double()[:,None]),-1).softmax(-1)[:,:-1]
                    truth=torch.einsum('ht,htd->hd',probability,value).float()
                    torch.testing.assert_close(actual[index,:,0].cpu(),truth,rtol=4e-5,atol=2e-6)
                    errors.append(float((actual[index,:,0].cpu()-truth).abs().max()))
                samples={'baseline':[],'candidate':[]};peaks={}
                for round_index in range(8):
                    for name,fn in ([('baseline',baseline),('candidate',candidate)] if round_index%2==0 else [('candidate',candidate),('baseline',baseline)]):
                        torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record();result=fn();end.record();end.synchronize()
                        if round_index>=2:samples[name].append(start.elapsed_time(end))
                        peaks[name]=torch.cuda.max_memory_allocated()-before;del result
                row={'tokens':count,'window':window,'fp64_max_abs':max(errors),'baseline_max_abs':float((actual-expected).abs().max()),
                    'milliseconds':{k:statistics.median(v) for k,v in samples.items()},'samples':samples,'incremental_peak_bytes':peaks}
                rows.append(row);print(json.dumps(row),flush=True)
                Path(args.output).write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'bounded':args.bounded or args.production,'production':args.production,'rows':rows},indent=2)+'\n')
                del actual,expected


if __name__=='__main__':main()

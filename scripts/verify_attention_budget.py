"""A2 prefill attention memory bound, exact partition equivalence and FP64 checks."""
import argparse
import json
from pathlib import Path
import os
import torch
from dllm.kernels import attention


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.manual_seed(41);rows=[];budget=32*1024**2
    with torch.inference_mode():
        for count,context in ((513,1024),(4096,4096)):
            q=torch.randn(count,64,1,64,device='cuda')
            k=torch.randn(2,8,context,64,device='cuda');v=torch.randn_like(k)
            positions=torch.arange(count,device='cuda');slots=positions%2
            sinks=torch.linspace(-100,100,64,device='cuda')
            for window in (0,128):
                def run(limit):
                    torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    output=attention(q,k,v,positions,slots,window=window,sinks=sinks,scratch_bytes=limit)
                    torch.cuda.synchronize();return output,torch.cuda.max_memory_allocated()-before
                whole,unbounded=run(None);bounded,peak=run(budget)
                assert torch.equal(whole,bounded)
                # Full output is linear in prompt length; bound is for temporary
                # partial/LSE storage plus the temporary chunk output.
                assert peak<=bounded.numel()*bounded.element_size()+budget+512*1024
                errors=[]
                for index in (0,1,125,126,127,128,count-1):
                    query=q[index,:,0].cpu().double()
                    key=k[index%2].cpu().double().repeat_interleave(8,dim=0)
                    value=v[index%2].cpu().double().repeat_interleave(8,dim=0)
                    score=torch.einsum('hd,htd->ht',query,key)*64**-.5
                    valid=torch.arange(context)<=index
                    if window:valid &= torch.arange(context)>index-window
                    score.masked_fill_(~valid[None,:],-float('inf'))
                    probability=torch.cat((score,sinks.cpu().double()[:,None]),dim=-1).softmax(-1)[:,:-1]
                    expected=torch.einsum('ht,htd->hd',probability,value).float()
                    actual=bounded[index,:,0].cpu()
                    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
                    errors.append(float((actual-expected).abs().max()))
                rows.append({'queries':count,'context':context,'window':window,'heads':64,'kv_heads':8,'dim':64,
                    'two_slots':True,'exact_partition_equivalence':True,'fp64_max_absolute_error':max(errors),
                    'unbounded_incremental_peak_bytes':unbounded,'bounded_incremental_peak_bytes':peak,
                    'output_bytes':bounded.numel()*bounded.element_size(),'scratch_budget_bytes':budget})
                print(json.dumps(rows[-1]),flush=True);del whole,bounded
    report={'passed':True,'gpu':torch.cuda.get_device_name(),'rows':rows,
        'scope':'Memory and numerical verification only, not a timing benchmark.'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':args.output}))


if __name__=='__main__':main()

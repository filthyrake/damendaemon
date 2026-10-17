"""Compare sequential and parallel causal rows from identical CUDA KV state."""
import argparse
import json
import os

import torch
from damendaemon.runtime import Runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--width',type=int,default=2)
    p.add_argument('--carry-state',choices=('sequential','parallel'),default='sequential')
    args=p.parse_args()
    if args.width not in (2,4,8):p.error('width must be 2, 4 or 8')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DAMENDAEMON_DEVICE']='cuda';os.environ['DAMENDAEMON_SPECULATIVE_TOKENS']='0';torch.set_num_threads(4)
    runtime=Runtime({'a':args.model},context=256,batch_size=4)
    report={'model':args.model,'width':args.width,'carry_state':args.carry_state,'blocks':[]}
    try:
        with runtime.session('a') as (executor,_,_),torch.inference_mode():
            prompt=executor.model.chat_tokens([{'role':'user','content':'Repeat red blue green ten times.'}])
            first=int(executor.prefill(prompt,1).argmax(-1).item());expected=[first]
            while len(expected)<32:expected.append(int(executor.decode_batch([expected[-1]],[1])[1].item()))
            executor.prefill(prompt,1)
            linear=executor.linear;traces=[]
            def traced(x,w,bias=None):
                y=linear(x,w,bias);traces.append(y[0].clone());return y
            executor.linear=traced
            report.update(compute_dtype=str(executor.compute_dtype),residual_dtype=str(executor.residual_dtype),
                          llama_precision=executor.llama_precision,expected=expected)
            for offset in range(0,len(expected)-args.width,args.width):
                start=len(prompt)+offset
                saved=[kv.clone() for kv in executor.kv]
                inputs=expected[offset:offset+args.width]
                sequential=[];first_trace=None
                for row,token in enumerate(inputs):
                    traces.clear()
                    logits=executor.forward_batch(torch.tensor([token],device='cuda'),
                        torch.tensor([start+row],device='cuda'),torch.tensor([1],device='cuda'))
                    sequential.append(logits[0].clone())
                    if row==0:first_trace=list(traces)
                after=[kv[:,1,:,:start+args.width].clone() for kv in executor.kv]
                for kv,old in zip(executor.kv,saved):kv.copy_(old)
                traces.clear()
                parallel=executor.forward_batch(torch.tensor(inputs,device='cuda'),
                    torch.arange(start,start+args.width,device='cuda'),torch.ones(args.width,device='cuda',dtype=torch.long))
                sequential=torch.stack(sequential)
                single_ids=sequential.argmax(-1).tolist();parallel_ids=parallel.argmax(-1).tolist()
                def top(logits):
                    values,ids=logits.float().topk(3);return {'ids':ids.tolist(),'logits':values.tolist()}
                block={'offset':offset,'start':start,'baseline_ids':expected[offset+1:offset+args.width+1],'sequential_ids':single_ids,'parallel_ids':parallel_ids,
                    'max_logit_error':float((sequential.float()-parallel.float()).abs().max()),
                    'sequential_top':top(sequential),'parallel_top':top(parallel),
                    'linear_first_row_max_errors':[float((a.float()-b.float()).abs().max()) for a,b in zip(first_trace,traces)],
                    'kv_max_errors':[float((kv[:,1,:,:start+args.width].float()-old.float()).abs().max()) for kv,old in zip(executor.kv,after)]}
                report['blocks'].append(block)
                # Each comparison starts from identical KV; this chooses which
                # path's accumulated rounding reaches the next comparison.
                if args.carry_state=='sequential':
                    for kv,old in zip(executor.kv,after):kv[:,1,:,:start+args.width].copy_(old)
                if single_ids!=parallel_ids or parallel_ids!=block['baseline_ids']:break
            report['same_argmax']=all(x['sequential_ids']==x['parallel_ids'] for x in report['blocks'])
            print(json.dumps(report,indent=2))
    finally:runtime.manage('unload',model='a',release_host=True)


if __name__=='__main__':main()

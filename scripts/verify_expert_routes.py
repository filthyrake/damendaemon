"""A2 stable-route permutation checks and full-model bit-exact prefill/decode A/B."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
import torch
from damendaemon.expert_routes import grouped_routes
from damendaemon.qwen2 import ModelWeights
from damendaemon.gpt_oss import GptOssExecutor


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.manual_seed(53);rows=[]
    with torch.inference_mode():
        for count in (1,16,129,4096):
            for distribution in ('random','one','ends'):
                selected=torch.randint(0,32,(count,4),device='cuda')
                if distribution=='one':selected.zero_()
                if distribution=='ends':selected=(selected%2)*31
                groups=list(grouped_routes(selected,32));expected=[]
                for expert in range(32):
                    tokens,routes=torch.where(selected==expert)
                    if tokens.numel():expected.append((expert,tokens,tokens*4+routes))
                assert len(groups)==len(expected)
                for (a,t,i),(b,bt,bi) in zip(groups,expected):assert a==b and torch.equal(t,bt) and torch.equal(i,bi)
                rows.append({'kind':'permutation','tokens':count,'distribution':distribution,'bit_exact':True})
        del groups,expected,selected
        model=ModelWeights(args.model,4096);executor=GptOssExecutor(model,max_batch=2)
        base=model.raw_tokens('The capital of France is Paris. Write a short explanation of how the ocean affects coastal weather. ')
        for count in (16,129,513,789,4000):
            ids=(base*((count+len(base)-1)//len(base)))[:count]
            executor.sorted_prefill_routes=False;expected=executor.prefill(ids,0).cpu()
            baseline=[]
            for _ in range(8):
                token=int(expected.argmax(-1)[0]);baseline.append(token)
                expected,_=executor.decode_batch([token],[0]);expected=expected.cpu()
            executor.sorted_prefill_routes=False;expected=executor.prefill(ids,0).cpu()
            executor.sorted_prefill_routes=True;actual=executor.prefill(ids,1).cpu()
            assert torch.equal(actual,expected),(count,float((actual-expected).abs().max()))
            generated=[]
            for token in baseline:
                assert int(actual.argmax(-1)[0])==token
                generated.append(token);actual,_=executor.decode_batch([token],[1]);actual=actual.cpu()
            rows.append({'kind':'model','tokens':count,'different_slots':True,'prefill_bit_exact':True,'greedy_tokens_exact':generated})
            print(json.dumps(rows[-1]),flush=True)
    Path(args.output).write_text(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()

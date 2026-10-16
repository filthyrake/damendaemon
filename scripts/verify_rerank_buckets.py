"""Mixed-length reranking: CPU reference, original-order proof and GPU timing."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from dllm.encoder import EncoderExecutor,EncoderWeights
from dllm.reranking import length_groups
from dllm.device import device_name
from dllm.runtime import require_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',required=True,choices=('cuda','mps'));args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device;require_gpu();torch.set_num_threads(4)
    if args.device=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    model=EncoderWeights(args.model,512);executor=EncoderExecutor(model)
    reference=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager',trust_remote_code=False).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    query='hello';docs=['blue '*n for n in (20,100,220,508)]
    cases={'uniform_short':[0]*64,'one_long':[0]*31+[3]+[0]*32,
        'one_medium':[2]+[0]*63,'four_lengths':[3,0,1,0,2,0,0,0]*8,
        'uniform_long':[3]*16}
    def sync():torch.cuda.synchronize() if args.device=='cuda' else torch.mps.synchronize()
    def measure(call):
        sync();start=time.perf_counter();scores=call();sync()
        return scores,(time.perf_counter()-start)*1000
    rows=[]
    with torch.inference_mode():
        expected=torch.cat([reference(**tokenizer(query,doc,return_tensors='pt')).logits.reshape(-1) for doc in docs])
        for name,indices in cases.items():
            tokens,types=model.encode_pairs(query,[docs[i] for i in indices])
            scores,cold=measure(lambda:executor.rerank(tokens,types))
            want=expected[indices]
            torch.testing.assert_close(scores.cpu(),want,atol=.0002,rtol=.0002)
            baseline,_=measure(lambda:executor._run(tokens,types).reshape(-1))
            torch.testing.assert_close(scores,baseline,atol=.0002,rtol=.0002)
            saved=scores.clone();keys=list(executor.graphs)
            other=executor.rerank(list(reversed(tokens)),list(reversed(types)))
            torch.testing.assert_close(other,scores.flip(0),atol=.0002,rtol=.0002)
            torch.testing.assert_close(scores,saved,atol=0,rtol=0)
            if length_groups(tokens) is not None:assert list(executor.graphs)==keys
            eager=[];bucket=[]
            # Alternate order to reduce warm-clock/order bias.
            for step in range(6):
                functions=[(eager,lambda:executor._run(tokens,types).reshape(-1)),
                    (bucket,lambda:executor.rerank(tokens,types))]
                for target,call in functions[::1 if step%2==0 else -1]:target.append(measure(call)[1])
            groups=length_groups(tokens)
            rows.append({'case':name,'batch':len(tokens),'lengths':list(map(len,tokens)),
                'groups':groups,'padded_tokens_before':len(tokens)*max(map(len,tokens)),
                'padded_tokens_after':sum(len(g)*max(len(tokens[i]) for i in g) for g in groups) if groups else len(tokens)*max(map(len,tokens)),
                'first_request_ms':cold,'baseline_ms':eager,'bucketed_ms':bucket,
                'median_baseline_ms':statistics.median(eager),'median_bucketed_ms':statistics.median(bucket),
                'reference_max_error':(scores.cpu()-want).abs().max().item(),
                'baseline_max_error':(scores-baseline).abs().max().item()})
    print(json.dumps({'passed':True,'model':args.model,'device':device_name(),'precision':'float32',
        'reference':'CPU FP32 Transformers eager, unique pair scores expanded to batch order',
        'returned_buffer_independent':True,'reverse_order_checked':True,'bucketed_graphs_not_captured':True,'rows':rows},indent=2))


if __name__=='__main__':main()

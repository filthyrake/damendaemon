"""Pinned multilingual XLM-RoBERTa CPU-reference qualification on real GPUs."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from dllm.encoder import EncoderWeights,EncoderExecutor
from dllm.device import device_name
from dllm.runtime import require_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',required=True,choices=('cuda','mps'));args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device;require_gpu();torch.set_num_threads(4)
    if args.device=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    model=EncoderWeights(args.model,2048);executor=EncoderExecutor(model)
    reference=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager',trust_remote_code=False).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    checks=[]
    @torch.inference_mode()
    def check(query,docs):
        tokens,types=model.encode_pairs(query,docs)
        batch=tokenizer([query]*len(docs),docs,padding=True,return_tensors='pt')
        for i,row in enumerate(tokens):
            assert batch['input_ids'][i,:len(row)].tolist()==row
            assert types[i]==[0]*len(row)
        expected=reference(**batch).logits.reshape(-1)
        start=time.perf_counter();actual=executor.rerank(tokens,types).cpu();elapsed=time.perf_counter()-start
        torch.testing.assert_close(actual,expected,atol=.0002,rtol=.0002)
        # Repeat with live graph buffers, then retain a caller-owned result.
        repeated=executor.rerank(tokens,types).cpu()
        torch.testing.assert_close(actual,repeated,atol=0,rtol=0)
        checks.append({'query':query,'documents':docs if max(map(len,docs))<256 else ['<long input>']*len(docs),
            'lengths':list(map(len,tokens)),'reference':expected.tolist(),'actual':actual.tolist(),
            'max_error':(actual-expected).abs().max().item(),'first_request_seconds':elapsed})
        print('checked '+str(len(checks)),file=__import__('sys').stderr,flush=True)
        return actual
    query='What is the capital of France?';docs=['Paris is the capital of France.',
        'Berlin is the capital of Germany.','A puppy is playing in a garden.']
    first=check(query,docs);assert first[0]>first[1] and first[0]>first[2]
    singleton=torch.cat([check(query,[doc]) for doc in docs]);torch.testing.assert_close(first,singleton,atol=.0002,rtol=.0002)
    check('法国的首都是哪里？',['法国的首都是巴黎。','柏林是德国的首都。','一只小狗在花园里玩耍。'])
    check('Quelle est la capitale de la France ?',['Paris est la capitale de la France.','Berlin ist die Hauptstadt von Deutschland.'])
    check('日本の首都はどこですか？',['日本の首都は東京です。','باريس هي عاصمة فرنسا.','café 🦙 — Zürich'])
    # Literal pad tokens inside an attended text exercise skipped position IDs.
    pad_docs=['one <pad> two','one two <pad> three','<pad> <pad> hello']
    tokens,_=model.encode_pairs('hello',pad_docs);assert any(model.config['pad_token_id'] in row for row in tokens)
    check('hello',pad_docs)
    overhead=len(tokenizer('hello','blue')['input_ids'])-1
    for length in (512,1024,2048):
        text='blue '*(length-overhead)
        assert len(model.encode_pairs('hello',[text])[0][0])==length
        check('hello',[text])
    # Mixed padding, order and a max-context input in the same request.
    check('hello',['blue '*(512-overhead),'blue '*20,'blue '*100])
    # Exercise multiple long-context buckets and restoration to caller order.
    mixed=['blue '*(n-overhead) for n in (1024,26,512,26,256,26,26,26)]
    check('hello',mixed);check('hello',list(reversed(mixed)))
    try:model.encode_pairs('hello',['blue '*(2049-overhead)]);raise AssertionError('overlength accepted')
    except ValueError:pass
    assert model.config['embedding_max_length']==8192
    print(json.dumps({'passed':True,'model':args.model,'device':device_name(),'compute_dtype':str(executor.compute_dtype),
        'host_weight_bytes':model.bytes,'checkpoint_context':8192,'qualified_context':model.max_context,
        'reference':'CPU FP32 Transformers eager','pair_ids_equal':True,'internal_padding_positions_checked':True,
        'checks':checks},indent=2))


if __name__=='__main__':main()

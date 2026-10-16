"""Independent CPU FP32 reference for native GPU cross-encoder scoring."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from dllm.encoder import EncoderWeights,EncoderExecutor
from dllm.device import device_name
from dllm.runtime import require_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',required=True,choices=('cuda','mps'));args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device;require_gpu();torch.set_num_threads(4)
    model=EncoderWeights(args.model,1024);executor=EncoderExecutor(model)
    reference=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager',trust_remote_code=False).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    checks=[]
    @torch.inference_mode()
    def check(query,docs):
        tokens,types=model.encode_pairs(query,docs)
        expected_tokens=tokenizer([query]*len(docs),docs,padding=True,return_tensors='pt')
        for i,row in enumerate(tokens):
            assert expected_tokens['input_ids'][i,:len(row)].tolist()==row
            assert expected_tokens['token_type_ids'][i,:len(row)].tolist()==types[i]
        expected=reference(**expected_tokens).logits.reshape(-1)
        actual=executor.rerank(tokens,types).cpu()
        torch.testing.assert_close(actual,expected,atol=.0002,rtol=.0002)
        checks.append({'query':query,'lengths':list(map(len,tokens)),'reference':expected.tolist(),
            'actual':actual.tolist(),'max_error':(actual-expected).abs().max().item()})
        return actual
    query='How many people live in Berlin?'
    docs=['Berlin had a population of 3,520,031 in 2015.','Berlin is well known for its museums.',
        'Paris is the capital of France.','こんにちは — café 🦙',' ', 'blue sky '*40]
    singles=torch.cat([check(query,[doc]) for doc in docs]);mixed=check(query,docs)
    torch.testing.assert_close(mixed,singles,atol=.0002,rtol=.0002)
    assert mixed[0]>mixed[1] and mixed[0]>mixed[2]
    check('こんにちは — café 🦙',list(reversed(docs)))
    check('hello',['blue '*508])
    # Same padded shape, but different segment boundaries and mask contents.
    check('one two three',['four five six seven']);check('one',['two three four five six seven'])
    for length in (3,7,15,31,63,127):
        check('hello',['blue '*length]);assert len(executor.graphs)<=4
    if executor.use_graph:
        tokens,types=model.encode_pairs('one',['two three four five six seven'])
        retained=executor.rerank(tokens,types);saved=retained.clone()
        other,segments=model.encode_pairs('one two three',['four five six seven'])
        executor.rerank(other,segments)
        torch.testing.assert_close(retained,saved,atol=0,rtol=0)
    for method in (lambda:model.encode_pairs('hello',['blue '*509]),lambda:model.encode(['embedding']),
            lambda:executor.embed([[1,2,3]])):
        try:method();raise AssertionError('invalid task or overlength accepted')
        except ValueError:pass
    print(json.dumps({'passed':True,'model':args.model,'device':device_name(),
        'compute_dtype':str(executor.compute_dtype),'reference':'CPU FP32 Transformers eager',
        'max_input_tokens':model.max_context,'host_weight_bytes':model.bytes,'semantic_ordering':True,
        'pair_tokenization_equal':True,'checks':checks},indent=2))


if __name__=='__main__':main()

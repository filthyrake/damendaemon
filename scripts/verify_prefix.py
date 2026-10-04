"""Independent prefix-cache numerical qualification and slot/cache isolation."""
import argparse
import gc
import json
import os
import torch
from transformers import AutoTokenizer
from reference_model import load_reference
from dllm.qwen2 import ModelWeights
from dllm.runtime import require_gpu,make_executor
from dllm.prefix import PrefixCache
from dllm.device import device_name


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--reference-quantization',choices=('none','q8_0','q4_0'),default='none')
    args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device
    require_gpu();torch.set_num_threads(4)
    model=ModelWeights(args.model,1024)
    if model.quantization!=args.reference_quantization:
        raise ValueError('runtime and reference quantization must match')
    executor=make_executor(model,max_batch=4)
    reference=load_reference(args.model,reference_quantization=args.reference_quantization,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float32,attn_implementation='eager').eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    corpus=tokenizer.encode('A blue bird flies above the forest. '*100,add_special_tokens=False)
    cache=PrefixCache(executor,512)
    checks=[]
    def check(prompt,slot):
        reused,logits=cache.restore(prompt,slot)
        if logits is None:
            for start in range(reused,len(prompt),127):
                logits=executor.prefill_chunk(prompt[start:start+127],slot,start,final=start+127>=len(prompt))
            cache.remember(prompt,slot,logits)
        target=reference(torch.tensor([prompt]),use_cache=True)
        def compare(actual,wanted):
            actual,wanted=actual.float().cpu(),wanted.float().cpu()
            error=actual-wanted
            relative=(error.square().mean().sqrt()/wanted.square().mean().sqrt()).item()
            absolute=error.abs().max().item()
            assert relative<.01 and absolute<.5 and actual.argmax()==wanted.argmax(),(relative,absolute)
            checks.append({'relative_rms':relative,'max_absolute':absolute,'cached_tokens':reused,'prompt_tokens':len(prompt)})
        compare(logits,target.logits[:,-1])
        # Decode modifies this slot after its prefix was snapshotted.
        for _ in range(3):
            token=target.logits[:,-1].argmax().item()
            logits,_=executor.decode_batch([token],[slot])
            target=reference(torch.tensor([[token]]),past_key_values=target.past_key_values,use_cache=True)
            compare(logits,target.logits[:,-1])
        assert cache.tokens<=512 and len(cache.entries)<=4
        return reused
    prompt=corpus[:255]
    assert check(prompt,3)==0
    assert check(prompt,0)==255
    assert check(prompt+[1,2,3],2)==255
    assert check(prompt[:127],1)==126
    for offset in (20,50,80,110,140):check([offset]+corpus[:128],offset%4)
    assert check(prompt,0)<255
    check(corpus[:600],2)
    assert check(corpus[:600],1)==512
    print(json.dumps({'passed':True,'device':device_name(),'model':args.model,'checks':checks,
        'weight_quantization':model.quantization,'kernel_backend':executor.kernel_backend,
        'exact_and_divergent_prefixes':True,'slot_and_output_isolation':True,'bounded_eviction':True,
        'cache_token_budget':512,'entry_limit':4,'reference_dtype':'float32','reference_attention':'eager'},indent=2))


if __name__=='__main__':main()

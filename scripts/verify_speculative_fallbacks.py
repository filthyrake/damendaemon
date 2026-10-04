"""Speculation leaves unsupported memory/device scheduling paths unchanged."""
import argparse
import json
import os
import torch
from dllm.scheduler import BatchedRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',required=True,choices=('cuda','mps'));args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device;os.environ['DLLM_SPECULATIVE_TOKENS']='8';torch.set_num_threads(4)
    if args.device=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    cases=[('metal',{},None,'none')] if args.device=='mps' else [
        ('paged',{},'256','none'),('chunked',{'prefill_chunk_size':8},None,'none'),
        ('prefix',{'prefix_cache_tokens':64},None,'none'),('quantized',{},None,'q8_0'),
        ('graphs_disabled',{'use_graph':False},None,'none')]
    rows=[]
    for name,kwargs,pages,quant in cases:
        os.environ.pop('DLLM_KV_CACHE_TOKENS',None)
        if pages:os.environ['DLLM_KV_CACHE_TOKENS']=pages
        os.environ['DLLM_QUANTIZATION']=quant
        runtime=BatchedRuntime({'a':args.model},context=128,batch_size=2,host_models=1,host_gib=8,**kwargs)
        try:
            def request():
                events=list(runtime.generate('a',None,16,raw_prompt=[40,298,1111,13]*4))
                return [e['token'] for e in events if 'token' in e],events[-1]
            first,final=request();second,other=request();assert first==second
            assert final['timing']['speculative_blocks']==other['timing']['speculative_blocks']==0
            with runtime.runtime.session('a') as (executor,_,_):
                present=getattr(executor,'speculative_verifier',None) is not None
                if name in ('paged','quantized','metal','graphs_disabled'):assert not present
            del executor
            rows.append({'case':name,'verifier_allocated':present,'zero_speculative_blocks':True,
                'repeated_tokens_equal':True,'cached_prompt_tokens':other['timing']['cached_prompt_tokens']})
            runtime.manage('unload',model='a',release_host=True)
        finally:runtime.close()
    print(json.dumps({'passed':True,'device':args.device,'rows':rows},indent=2))


if __name__=='__main__':main()

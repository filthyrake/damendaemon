"""Real CUDA resident/peak allocation after load and representative generation."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import time
import torch
from damendaemon.runtime import require_gpu,make_executor
from damendaemon.qwen2 import ModelWeights,weight_bytes


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),default='none')
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--prompt-tokens',type=int,default=1024)
    args=p.parse_args()
    if not 1<=args.prompt_tokens<args.context:p.error('prompt must leave room for generation')
    require_gpu()
    if not torch.cuda.is_available():raise RuntimeError('CUDA memory measurement required')
    torch.set_num_threads(4)
    start=time.perf_counter()
    model=ModelWeights(args.path,args.context,quantization=args.quantization)
    host_seconds=time.perf_counter()-start
    assert model.bytes==weight_bytes(args.path,quantization=args.quantization)
    device_start=time.perf_counter()
    executor=make_executor(model,max_batch=4)
    torch.cuda.synchronize()
    device_seconds=time.perf_counter()-device_start
    resident=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ids=model.raw_tokens('A simple sentence about computers and memory. ')
    prompt=(ids*((args.prompt_tokens+len(ids)-1)//len(ids)))[:args.prompt_tokens]
    logits=executor.prefill_batch([prompt],[0])
    generated=[]
    for _ in range(min(8,args.context-len(prompt))):
        token=logits.argmax(-1).tolist()
        generated.extend(token)
        logits,_=executor.decode_batch(token,[0])
    torch.cuda.synchronize()
    print(json.dumps({'model':args.path,'quantization':args.quantization,'device':torch.cuda.get_device_name(),
        'torch':str(torch.__version__),'context':args.context,'batch_capacity':4,'prompt_tokens':len(prompt),
        'kv_layout':'paged' if hasattr(executor,'pool') else 'fixed',
        'kv_storage_bytes':sum(t.numel()*t.element_size() for t in executor.kv),
        'kv_token_capacity':getattr(executor,'kv_token_capacity',executor.context*executor.max_batch),
        'weight_cache_status':model.weight_cache_status,'generated_tokens':generated,
        'device_load_seconds':device_seconds,'load_and_generation_seconds':time.perf_counter()-start,
        'host_weight_bytes':model.bytes,'host_load_seconds':host_seconds,'resident_allocated_bytes':resident,
        'generation_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'generation_end_allocated_bytes':torch.cuda.memory_allocated(),
        'reserved_bytes':torch.cuda.memory_reserved(),
        'scope':'PyTorch live/peak allocations, not total process or driver memory; cold load includes conversion'},indent=2))


if __name__=='__main__':main()

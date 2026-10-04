"""Native GPT-OSS checkpoint smoke probe; not a Harmony qualification."""
import argparse
import json
import os
import time
import torch
from dllm.qwen2 import ModelWeights
from dllm.gpt_oss import GptOssExecutor


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--graph',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4)
    start=time.perf_counter();model=ModelWeights(args.model,256)
    print(json.dumps({'phase':'loaded_host','bytes':model.bytes,'seconds':time.perf_counter()-start}),flush=True)
    executor=GptOssExecutor(model,use_graph=args.graph,max_batch=1)
    prompt=model.tokenizer.encode('The capital of France is',add_special_tokens=False).ids
    start=time.perf_counter();logits=executor.prefill(prompt);torch.cuda.synchronize()
    prefill=time.perf_counter()-start;tokens=[];times=[]
    for step in range(8):
        token=int(logits.argmax(-1));tokens.append(token)
        start=time.perf_counter();logits,_=executor.decode_batch([token],[0]);torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    print(json.dumps({'phase':'generated','graph':args.graph,'gpu':torch.cuda.get_device_name(),
        'prompt_ids':prompt,'token_ids':tokens,'text':model.tokenizer.decode(tokens),
        'prefill_seconds':prefill,'decode_seconds':times,'allocated_bytes':torch.cuda.memory_allocated(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2),flush=True)


if __name__=='__main__':main()

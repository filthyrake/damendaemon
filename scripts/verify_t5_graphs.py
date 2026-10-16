"""CUDA prompt graph input replacement, slot isolation, eviction and cold cost."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import time
import torch
from transformers import AutoModelForSeq2SeqLM
from dllm.models import load
from dllm.runtime import make_executor,require_gpu


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    os.environ['DLLM_T5_PREFILL_GRAPHS']='2';require_gpu();torch.set_num_threads(4)
    model=load(args.model,512);executor=make_executor(model,max_batch=4)
    reference=AutoModelForSeq2SeqLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float32).eval()
    corpus=model.raw_tokens('summarize: The blue bird visits a green field while a dog sleeps. '*100)
    trials=[]
    cases=[([31],[0]),([31],[0]),([31],[0]),([15,32],[3,1]),([15,32],[3,1]),
           ([512],[2]),([31],[0]),([31],[0]),([15,32],[3,1])]
    base=torch.cuda.memory_allocated()
    for index,(lengths,slots) in enumerate(cases):
        prompts=[corpus[index:index+n] for n in lengths]
        old_hits=executor.prefill_graph_hits
        torch.cuda.synchronize();start=time.perf_counter()
        actual=executor.prefill_batch(prompts,slots);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        for row,prompt in enumerate(prompts):
            expected=reference(input_ids=torch.tensor([prompt]),decoder_input_ids=torch.tensor([[model.config['decoder_start_token_id']]])).logits[0,-1]
            value=actual[row].cpu();relative=((value-expected).square().mean()/expected.square().mean()).sqrt().item()
            assert relative<.001 and value.argmax()==expected.argmax(),relative
        assert len(executor.prefill_graphs)<=2
        trials.append({'lengths':lengths,'slots':slots,'reused':executor.prefill_graph_hits>old_hits,
            'seconds':elapsed,'relative_rms':relative,'graph_count':len(executor.prefill_graphs),
            'allocated_bytes_above_decode':torch.cuda.memory_allocated()-base})
        executor.decode_batch([8]*len(slots),slots)
    assert executor.prefill_graph_hits==4
    # This checkpoint's two-entry cache needs <16 MiB above decode. Repeated
    # creation of warmup streams leaked one retained 8-MiB cuBLAS workspace
    # per shape even after LRU eviction; this gate detects that regression.
    assert max(x['allocated_bytes_above_decode'] for x in trials)<16*1024**2
    print(json.dumps({'passed':True,'cache_limit':2,'trials':trials},indent=2))


if __name__=='__main__':main()

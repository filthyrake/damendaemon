"""Native verifier widths, nonzero slots, forced rejection and graph lifetime."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import json
import os
import time
import weakref

import torch
from damendaemon.runtime import Runtime
from damendaemon.speculative import Verifier


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--context',type=int,default=256)
    p.add_argument('--prompt-tokens',type=int,default=0);args=p.parse_args()
    if args.prompt_tokens<0 or args.prompt_tokens+32>args.context:p.error('prompt and output must fit context')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DAMENDAEMON_DEVICE']='cuda';os.environ['DAMENDAEMON_SPECULATIVE_TOKENS']='0';torch.set_num_threads(4)
    runtime=Runtime({'a':args.model},context=args.context,batch_size=4)
    rows=[]
    @torch.inference_mode()
    def baseline(executor,prompt,slot):
        output=[int(executor.prefill(prompt,slot).argmax(-1).item())]
        while len(output)<32:output.append(int(executor.decode_batch([output[-1]],[slot])[1].item()))
        return output
    try:
        with runtime.session('a') as (executor,_,_):
            prompt=executor.model.chat_tokens([{'role':'user','content':'Repeat red blue green ten times.'}],
                enable_thinking=False if executor.c['model_type']=='qwen3' else None)
            if args.prompt_tokens:
                text=''.join(f'Record {i}: The parcel is in cabinet {i%11}, drawer {i%7}.\n' for i in range(args.prompt_tokens))
                prompt=executor.model.raw_tokens(text)[:args.prompt_tokens]
                assert len(prompt)==args.prompt_tokens
            for width in (2,4,8):
                torch.cuda.synchronize();before=torch.cuda.memory_allocated();started=time.perf_counter()
                verifier=Verifier(executor,width);torch.cuda.synchronize();setup=time.perf_counter()-started
                retained=torch.cuda.memory_allocated()-before
                executor.prefill([40,298,1111,13],0)
                sentinel=[kv[:,0].clone() for kv in executor.kv]
                for slot in (1,3):
                    expected=baseline(executor,prompt,slot)
                    for mode in ('accept','reject','mixed'):
                        output=[int(executor.prefill(prompt,slot).argmax(-1).item())];blocks=matched=0
                        while len(output)<32:
                            start=executor.lengths[slot];remaining=32-len(output)
                            if remaining<2 or start+width>executor.context:
                                output.append(int(executor.decode_batch([output[-1]],[slot])[1].item()));continue
                            drafts=list(expected[len(output):len(output)+min(width-1,remaining-1)])
                            if mode=='reject' or (mode=='mixed' and blocks%2):drafts[0]=(drafts[0]+1)%executor.c['vocab_size']
                            tokens,accepted=verifier.verify(output[-1],drafts,start,slot)
                            output.extend(tokens);executor.lengths[slot]=start+len(tokens)
                            blocks+=1;matched+=accepted
                        assert all(torch.equal(kv[:,0],saved) for kv,saved in zip(executor.kv,sentinel))
                        # Once greedy paths diverge, a baseline-derived wrong draft may match
                        # the new target path; report it instead of masking the original failure.
                        rows.append({'width':width,'slot':slot,'mode':mode,'exact_tokens':output==expected,
                            'expected':expected,'actual':output,
                            'first_mismatch':next((i for i,(a,b) in enumerate(zip(expected,output)) if a!=b),None),
                            'other_slot_unchanged':True,'blocks':blocks,'matched':matched,
                            'setup_seconds':setup,'retained_allocated_bytes':retained})
                graph_ref=weakref.ref(verifier.graph);ref=weakref.ref(verifier)
                del verifier;gc.collect();torch.cuda.synchronize()
                assert graph_ref() is None and ref() is None
            passed=all(row['exact_tokens'] and (row['mode']!='reject' or row['matched']==0) for row in rows)
            print(json.dumps({'passed':passed,'model':args.model,'gpu':'NVIDIA A2',
                'compute_dtype':str(executor.compute_dtype),'residual_dtype':str(executor.residual_dtype),
                'kv_dtype':str(executor.kv[0].dtype),'matrix_storage_dtype':str(executor.w['head'].dtype),
                'qwen3_precision':executor.qwen3_precision,'llama_precision':executor.llama_precision,
                'dense_precision':executor.dense_precision,'context':args.context,'prompt_tokens':len(prompt),
                'input_kind':'raw' if args.prompt_tokens else 'chat','graph_reclamation':True,'rows':rows},indent=2))
            if not passed:raise SystemExit(1)
    finally:runtime.manage('unload',model='a',release_host=True)


if __name__=='__main__':main()

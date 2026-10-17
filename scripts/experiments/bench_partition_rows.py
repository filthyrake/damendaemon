"""Matched model-level row-bound experiment; no HTTP/network throughput claim."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import signal
import time
import torch
from damendaemon.partition import PartitionExecutor
from damendaemon.qwen2 import ModelWeights
from damendaemon import q4_stage_kernels
from damendaemon import kernels
from scripts.experiments.q4_dynamic_rows import install_validation_dispatch


def sync():
    for device in range(2):torch.cuda.synchronize(device)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt');p.add_argument('--output',required=True);p.add_argument('--integrated',action='store_true')
    p.add_argument('--compare-rope',action='store_true',help='hold integrated row bounds on and vary only the experimental rotary stride')
    p.add_argument('--integrated-rope',action='store_true',help='use the integrated stage rotary policy in the rotary comparison')
    args=p.parse_args()
    if args.integrated_rope and not args.compare_rope:p.error('--integrated-rope requires --compare-rope')
    def terminate(number,frame):raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,terminate)
    assert torch.cuda.device_count()==2,'select exactly two GPUs'
    prompt=json.loads(Path(args.prompt).read_text());assert len(prompt)==16
    os.environ.update(DAMENDAEMON_Q4_STAGE_ADJACENT='1',DAMENDAEMON_PARTITION_PACKED_PREFILL='1')
    torch.set_num_threads(4)
    model=ModelWeights(args.model,512,quantization='q4_0')
    original=q4_stage_kernels.single_dot_adjacent
    original_rope=kernels._rope_store
    report=dict(passed=False,integrated=args.integrated,integrated_rope=args.integrated_rope,comparison='rotary-stride' if args.compare_rope else 'row-bounds',model=args.model,quantization='q4_0',context=512,capacity=4,prompt_tokens=16,output_tokens=32,
        devices=os.environ['CUDA_VISIBLE_DEVICES'],trials=[],
        scope='Same-host model execution ABBA. Static versus selective runtime row bounds only; adjacent weights and packed prefill fixed. Host weights loaded once; all group shapes warmed before steady-state timing. First-shape calls are observed per process/order, not an isolated cold-start comparison. No HTTP or network throughput claim.',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ('damendaemon/q4_stage.py','damendaemon/q4_stage_kernels.py','damendaemon/kernels.py','damendaemon/partition.py','damendaemon/layer_stage.py','damendaemon/qwen2.py','damendaemon/partition_policy.py',
                  'scripts/experiments/q4_dynamic_rows.py','scripts/experiments/rope_runtime_tokens.py',__file__)})
    if args.compare_rope:
        report['scope']='Same-host model execution ABBA. Static versus runtime rotary token stride only; integrated selective row bounds, adjacent weights and packed prefill fixed. All group shapes warmed. First-shape observations share process/cache state and are not isolated cold-start evidence. No HTTP or network throughput claim.'
    references={};first_references={}
    executor=None
    try:
        for dynamic in (False,True,True,False):
            q4_stage_kernels.single_dot_adjacent=original
            kernels._rope_store=original_rope
            os.environ['DAMENDAEMON_Q4_STAGE_RUNTIME_ROWS']=str(int(args.compare_rope or (dynamic and args.integrated)))
            os.environ['DAMENDAEMON_Q4_STAGE_RUNTIME_ROPE']=str(int(args.compare_rope and args.integrated_rope and dynamic))
            if args.compare_rope and dynamic and not args.integrated_rope:
                from scripts.experiments.rope_runtime_tokens import install_validation_dispatch as install_rope
                install_rope()
            elif dynamic and not args.integrated and not args.compare_rope:install_validation_dispatch()
            sync();started=time.perf_counter()
            executor=PartitionExecutor(model,12,use_graph=True,max_batch=4)
            sync();trial=dict(dynamic=dynamic,activation_seconds=time.perf_counter()-started,first_shapes=[],runs=[])
            report['trials'].append(trial)
            for length in (3,5,7,11,13,17,19,23,31,33,47,63,65,95,127):
                tokens=(prompt*((length+15)//16))[:length]
                sync();started=time.perf_counter();logits=executor.prefill_batch([tokens],[0]);sync()
                elapsed=time.perf_counter()-started;token=int(logits.argmax(-1).item())
                first_references.setdefault(length,token);assert token==first_references[length]
                trial['first_shapes'].append(dict(prompt_tokens=length,seconds=elapsed,token=token))
            for size in (1,2,3,4):executor.prefill_batch([prompt]*size,list(range(size)))
            sync()
            for size in (1,2,4):
                slots=list(range(size))
                for iteration in range(7):
                    sync();started=time.perf_counter()
                    logits=executor.prefill_batch([prompt]*size,slots)
                    tokens=logits.argmax(-1).tolist();generated=[tokens]
                    sync();first=time.perf_counter()
                    for _ in range(31):
                        logits,chosen=executor.decode_batch(tokens,slots)
                        tokens=chosen.tolist();generated.append(tokens)
                    sync();finished=time.perf_counter()
                    references.setdefault(size,generated);assert generated==references[size]
                    trial['runs'].append(dict(batch_size=size,warmup=iteration==0,first_token_ms=1000*(first-started),
                        total_ms=1000*(finished-started),output_tokens_per_second=32*size/(finished-started),
                        decode_tokens_per_second=31*size/(finished-first),tokens=generated))
            del logits,chosen
            executor.close();executor=None;gc.collect()
            for device in range(2):
                with torch.cuda.device(device):torch.cuda.empty_cache()
            print('dynamic',dynamic,'complete',flush=True)
        report['summary']=[]
        for size in (1,2,4):
            for dynamic in (False,True):
                samples=[r for t in report['trials'] if t['dynamic']==dynamic for r in t['runs']
                         if r['batch_size']==size and not r['warmup']]
                report['summary'].append(dict(dynamic=dynamic,batch_size=size,samples=len(samples),
                    **{key:statistics.median(r[key] for r in samples) for key in
                       ('first_token_ms','total_ms','output_tokens_per_second','decode_tokens_per_second')}))
        report['passed']=True
    finally:
        if executor is not None:executor.close()
        q4_stage_kernels.single_dot_adjacent=original
        kernels._rope_store=original_rope
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

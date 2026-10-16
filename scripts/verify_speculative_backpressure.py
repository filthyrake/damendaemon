"""Force a full two-block stream buffer and prove other requests still progress."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import torch
from dllm.scheduler import BatchedRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DLLM_DEVICE']='cuda';os.environ['DLLM_SPECULATIVE_TOKENS']='8';torch.set_num_threads(4)
    runtime=BatchedRuntime({'a':args.model},context=256,batch_size=4)
    messages=[{'role':'user','content':'Write the following sentence exactly ten times, with no introduction: The quick brown fox jumps over the lazy dog.'}]
    try:
        with runtime.runtime.session('a') as (executor,_,_):
            verifier=executor.speculative_verifier;executor.speculative_verifier=None
            options={'enable_thinking':False} if executor.c['model_type']=='qwen3' else {}
        def generate():return runtime.generate('a',messages,64,**options)
        expected=[e['token'] for e in generate() if 'token' in e]
        executor.speculative_verifier=verifier
        def wait_full():
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                with runtime.condition:requests=list(runtime.requests)
                for request in requests:
                    if request.events.full():
                        assert request.events.maxsize==16
                        return request
                time.sleep(.005)
            raise AssertionError('stream buffer did not fill')
        stream=generate();first=next(stream);request=wait_full();observed=request.events.qsize()
        with ThreadPoolExecutor(1) as pool:
            other=pool.submit(lambda:list(runtime.generate('a',[{'role':'user','content':'Say hello.'}],8,**options)))
            events=other.result(timeout=10);assert events[-1]['finish_reason'] in ('stop','length')
        tokens=[first['token']]+[e['token'] for e in stream if 'token' in e]
        assert tokens==expected
        cancelled=generate();next(cancelled);request=wait_full();cancelled.close()
        assert request.done.wait(timeout=3)
        assert [e['token'] for e in generate() if 'token' in e]==expected
        print(json.dumps({'passed':True,'model':args.model,'observed_full_queue':observed,
            'queue_limit':16,'other_request_progress_while_blocked':True,'exact_tokens_after_drain':True,
            'cancel_full_queue_recovers':True},indent=2))
    finally:runtime.close()


if __name__=='__main__':main()

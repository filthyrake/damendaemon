"""Scoring failure recovery and interleaving with ordinary captured generation."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import torch
from damendaemon.scheduler import BatchedRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',required=True);args=p.parse_args()
    os.environ['DAMENDAEMON_DEVICE']=args.device;torch.set_num_threads(4)
    runtime=BatchedRuntime({'a':args.model},context=256,batch_size=4,host_models=1,host_gib=8)
    prompt=[40,298,1111,13]*4
    def generate():
        return [e['token'] for e in runtime.generate('a',None,8,raw_prompt=prompt) if 'token' in e]
    try:
        baseline=generate()
        for ids in ([0,1],[1,1000000000],[2,9,22]):
            try:result=runtime.score('a',prompt,ids)
            except ValueError:
                assert ids==[1,1000000000]
            else:assert [x['token_id'] for x in result['data']]==ids
            assert generate()==baseline
        with ThreadPoolExecutor(4) as pool:
            jobs=[pool.submit(generate),pool.submit(runtime.score,'a',prompt,[0,1]),
                  pool.submit(generate),pool.submit(runtime.score,'a',prompt,[17,33])]
            values=[f.result() for f in jobs]
        assert values[0]==values[2]==baseline
        assert [x['token_id'] for x in values[1]['data']]==[0,1]
        assert [x['token_id'] for x in values[3]['data']]==[17,33]
        with runtime.runtime.session('a') as (executor,_,_):
            assert getattr(executor,'_candidate_head',None) is None
            if hasattr(executor,'pool'):assert not executor.reserved_slots
            backend=executor.kernel_backend
        print(json.dumps({'passed':True,'device':args.device,'backend':backend,'tokens':baseline,
            'failure_recovery':True,'generation_unchanged':True,'interleaved_requests':True},indent=2))
    finally:runtime.close()


if __name__=='__main__':main()

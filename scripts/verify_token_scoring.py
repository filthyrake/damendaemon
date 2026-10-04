"""Real GPU scoring correctness, request isolation and matched prefill timing."""
import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
import torch
from dllm.device import synchronize,device_name
from dllm.runtime import request_kv
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--other-model',required=True)
    p.add_argument('--device',required=True,choices=('cuda','mps'));args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device;torch.set_num_threads(4)
    runtime=BatchedRuntime({'a':args.model,'b':args.other_model},context=2048,batch_size=4,
        gpu_models=1,host_models=2,host_gib=16)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/token-scores'
    def request(alias='a',ids=None):
        body={'model':alias,'prompt':[40,298,1111,13]*4,'candidate_token_ids':ids or [0,1,100,1000]}
        with urlopen(Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:
            return json.load(r)
    def timed(fn):
        for _ in range(3):fn()
        synchronize();values=[]
        for _ in range(15):
            start=time.perf_counter();fn();synchronize();values.append((time.perf_counter()-start)*1000)
        return statistics.median(values)
    reports=[]
    try:
        for alias,path in [('a',args.model),('b',args.other_model)]:
            response=request(alias)
            assert response['normalization']=='candidate_set' and response['calibrated'] is False
            assert response['usage']['completion_tokens']==0
            with runtime.runtime.session(alias) as (executor,_,_):
                rows=[]
                for count in (16,256,1024):
                    prompt=([40,298,1111,13]*(count//4))
                    with request_kv(executor,count):
                        reference=executor.prefill(prompt).float()
                        for n in (2,32,256):
                            ids=[int(i) for i in torch.linspace(0,executor.c['vocab_size']-1,n).tolist()]
                            index=torch.tensor(ids,device=executor.device)
                            expected=reference.index_select(-1,index)
                            actual=executor.score_candidates(prompt,ids)
                            torch.testing.assert_close(actual,expected,atol=.00005,rtol=.00005)
                            torch.testing.assert_close(actual.softmax(-1),expected.softmax(-1),atol=.004,rtol=.02)
                            assert actual.argmax().item()==expected.argmax().item()
                            full=timed(lambda:executor.prefill(prompt).float().index_select(-1,index))
                            narrow=timed(lambda:executor.score_candidates(prompt,ids))
                            rows.append({'prompt_tokens':count,'candidates':n,'max_logit_error':(actual-expected).abs().max().item(),
                                'full_head_prefill_ms':full,'selected_head_prefill_ms':narrow,'speedup':full/narrow})
                with request_kv(executor,16):
                    logits=executor.prefill([40,298,1111,13]*4).float()[0,[0,1,100,1000]]
                torch.testing.assert_close(torch.tensor([item['probability'] for item in response['data']]),
                    logits.softmax(-1).cpu(),atol=.004,rtol=.02)
                assert getattr(executor,'_candidate_head',None) is None
            reports.append({'model':path,'rows':rows,'http_response':response})
        restored=request('a');assert restored['dllm_timing']['source_tier']=='host'
        sets=[[0,1],[17,33,65],[100,1000,2000],[7,19,101,999]]
        baseline=[request(ids=ids)['data'] for ids in sets]
        with ThreadPoolExecutor(4) as pool:outputs=list(pool.map(lambda ids:request(ids=ids),sets))
        assert [r['data'] for r in outputs]==baseline
        # Captured full-vocabulary decode still works after temporary scoring heads.
        generated=list(runtime.generate('a',None,4,raw_prompt=[40,298,1111,13]*4))
        assert generated[-1]['usage']['completion_tokens']==4
        print(json.dumps({'passed':True,'device':device_name(),'torch':torch.__version__,
            'reports':reports,'concurrent_isolation':True,'host_restore':True,'generation_after_scoring':True,
            'timing':'median of 15 synchronized eager prefills after 3 warmups; selected includes row extraction; full includes result gather; no HTTP or loading'},indent=2))
    finally:
        server.shutdown();server.server_close();worker.join();runtime.close()


if __name__=='__main__':main()

"""Matched direct-scoring HTTP timing with prefill graph reuse off and on."""
import argparse
import json
import os
import statistics
import threading
import time
from urllib.request import Request,urlopen
import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args();torch.set_num_threads(4)
    results=[];baseline={}
    for limit in (0,4):
        os.environ['DLLM_PREFILL_GRAPHS']=str(limit)
        runtime=BatchedRuntime({'a':args.model},context=256,batch_size=4,host_models=1,host_gib=8)
        server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            runtime.manage('load',model='a')
            for count in (16,128):
                body={'model':'a','prompt':[40]*count,'candidate_token_ids':[0,1,100,1000]}
                trials=[]
                for index in range(16):
                    start=time.perf_counter()
                    with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/token-scores',
                            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
                        value=json.load(response)
                    elapsed=(time.perf_counter()-start)*1000
                    scores=[x['logit'] for x in value['data']]
                    if limit==0:baseline[count]=scores
                    else:torch.testing.assert_close(torch.tensor(scores),torch.tensor(baseline[count]),atol=.00005,rtol=.00005)
                    trials.append({'warmup':index==0,'milliseconds':elapsed,'data':value['data']})
                results.append({'graph_limit':limit,'prompt_tokens':count,'candidate_ids':body['candidate_token_ids'],
                    'first_ms':trials[0]['milliseconds'],'warm_median_ms':statistics.median(t['milliseconds'] for t in trials[1:]),
                    'trials':trials})
        finally:
            server.shutdown();server.server_close();worker.join();runtime.close()
    print(json.dumps({'passed':True,'model':args.model,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'results':results,'timing':'single client loopback HTTP, models preloaded; first request kept separately, then 15 measured requests; no answer decoding'},indent=2))


if __name__=='__main__':main()

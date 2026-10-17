"""Real HTTP reranking, mixed-task residency, failure recovery and pair timing."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import torch
from damendaemon.scheduler import BatchedRuntime
from damendaemon.server import BoundedHTTPServer,handler_for


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--encoder',required=True);p.add_argument('--decoder',required=True)
    p.add_argument('--device',choices=('cuda','mps'),required=True);args=p.parse_args()
    os.environ['DAMENDAEMON_DEVICE']=args.device;torch.set_num_threads(4)
    runtime=BatchedRuntime({'rank':args.model,'embed':args.encoder,'chat':args.decoder},context=512,
        batch_size=4,gpu_models=1,host_models=3,host_gib=8)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    query='How many people live in Berlin?'
    docs=['Berlin had a population of 3,520,031 in 2015.','Berlin is well known for its museums.',
          'Paris is the capital of France.','A puppy is playing in a garden.']
    def post(path,body):
        with urlopen(Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:return json.load(r)
    def rank(documents=docs,**changes):
        return post('/v1/rerank',{'model':'rank','query':query,'documents':documents,**changes})
    try:
        first=rank(return_documents=True);assert first['results'][0]['index']==0
        assert first['score_type']=='raw_relevance' and first['calibrated'] is False
        with runtime.runtime.session('rank') as (executor,_,_):
            tokens,_=executor.model.encode_pairs(query,docs)
        assert first['usage']['prompt_tokens']==sum(map(len,tokens))
        limited=rank(top_n=2);assert limited['results']==[{k:v for k,v in r.items() if k!='document'} for r in first['results'][:2]]
        assert [r['document']['text'] for r in first['results']]==[docs[r['index']] for r in first['results']]
        embedded=post('/v1/embeddings',{'model':'embed','input':'A test sentence.'})
        assert len(embedded['data'][0]['embedding'])==384
        generated=post('/v1/completions',{'model':'chat','prompt':[40,298,1111,13]*4,'max_tokens':4,'temperature':0})
        restored=rank(return_documents=True);assert restored['results']==first['results']
        assert restored['damendaemon_timing']['source_tier']=='host'
        for path,body in [('/v1/rerank',{'model':'embed','query':query,'documents':docs}),
                ('/v1/embeddings',{'model':'rank','input':'text'}),
                ('/v1/completions',{'model':'rank','prompt':'text','max_tokens':4}),
                ('/v1/rerank',{'model':'rank','query':'hello','documents':['blue '*509]}),
                ('/v1/rerank',{'model':'rank','query':query,'documents':docs,'top_n':True})]:
            try:post(path,body);raise AssertionError('invalid request accepted')
            except HTTPError as error:
                with error:
                    if error.code != 400:
                        raise ValueError(error.read())
        assert rank(return_documents=True)['results']==first['results']
        sets=[docs,list(reversed(docs)),docs[1:]+docs[:1],[docs[0],docs[0],docs[1]]]
        expected=[rank(documents=items)['results'] for items in sets]
        with ThreadPoolExecutor(4) as pool:actual=list(pool.map(lambda items:rank(documents=items)['results'],sets))
        assert actual==expected
        rows=[]
        for count in (1,4,16,64):
            documents=(docs*16)[:count];trials=[]
            for i in range(11):
                start=time.perf_counter();response=rank(documents=documents);elapsed=time.perf_counter()-start
                trials.append({'warmup':i==0,'seconds':elapsed,'pairs_per_second':count/elapsed})
                assert len(response['results'])==count
            median=statistics.median(t['seconds'] for t in trials[1:])
            rows.append({'pairs':count,'median_ms':median*1000,'pairs_per_second':count/median,'trials':trials})
        print(json.dumps({'passed':True,'model':args.model,'device':args.device,'example':first,
            'host_restore':True,'encoder_decoder_switching':True,'task_errors_recover':True,
            'concurrent_isolation':True,'benchmarks':rows,
            'timing':'single HTTP client; 10 measured rounds after 1 warmup at each pair count; repeated short texts; no cross-engine comparison'},indent=2))
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()

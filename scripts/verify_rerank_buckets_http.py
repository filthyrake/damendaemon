"""Live HTTP A/B proof of mixed-length batching, including concurrent isolation."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import statistics
import threading
import time
from urllib.request import Request,urlopen

import torch
import dllm.reranking as reranking
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--device',required=True,choices=('cuda','mps'));args=p.parse_args()
    if args.device=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DLLM_DEVICE']=args.device;torch.set_num_threads(4)
    runtime=BatchedRuntime({'rank':args.model},context=512,batch_size=4)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/rerank'
    documents=['blue '*20]*31+['blue '*508]+['blue '*20]*32
    def request(docs):
        start=time.perf_counter()
        with urlopen(Request(url,data=json.dumps({'model':'rank','query':'hello','documents':docs,
                'return_documents':True}).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:
            result=json.load(r)
        elapsed=(time.perf_counter()-start)*1000
        assert sorted(item['index'] for item in result['results'])==list(range(len(docs)))
        assert all(item['document']['text']==docs[item['index']] for item in result['results'])
        assert result['usage']['prompt_tokens']==sum(4+len(doc.split()) for doc in docs)
        return result,elapsed
    original=reranking.length_groups
    try:
        reference,_=request(documents)
        baseline=[];optimized=[];max_error=0
        def compare(result,expected):
            a={item['index']:item['relevance_score'] for item in result['results']}
            b={item['index']:item['relevance_score'] for item in expected['results']}
            assert a.keys()==b.keys() and result['usage']==expected['usage']
            error=max(abs(a[i]-b[i]) for i in a);assert error<.0002
            return error
        for step in range(6):
            # No request is in flight when the experimental baseline hook changes.
            # Disable only partitioning to reproduce the original padded path.
            modes=[(baseline,lambda tokens:None),(optimized,original)]
            for output,selector in modes[::1 if step%2==0 else -1]:
                reranking.length_groups=selector
                result,elapsed=request(documents);output.append(elapsed)
                max_error=max(max_error,compare(result,reference))
        reranking.length_groups=original
        sets=[documents,list(reversed(documents)),documents[16:]+documents[:16],documents[:16]]
        expected=[request(docs)[0] for docs in sets]
        with ThreadPoolExecutor(4) as pool:actual=list(pool.map(request,sets))
        for (result,_),want in zip(actual,expected):max_error=max(max_error,compare(result,want))
        print(json.dumps({'passed':True,'device':args.device,'model':args.model,'precision':'float32',
            'baseline':'Same server, length partitioning disabled between completed requests',
            'baseline_ms':baseline,'optimized_ms':optimized,'median_baseline_ms':statistics.median(baseline),
            'median_optimized_ms':statistics.median(optimized),'max_score_error':max_error,
            'usage':reference['usage'],'concurrent_isolation':True,'document_mapping':True,
            'workload':'63 pairs of 24 tokens, one pair of 512 tokens; six alternating-order measured rounds'},indent=2))
    finally:
        reranking.length_groups=original
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()

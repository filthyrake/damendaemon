"""Real A2 reranker graph replay, bounds, reclamation and latency proof."""
import argparse
import gc
import json
import os
import statistics
import time
import weakref

import torch
from dllm.encoder import EncoderExecutor,EncoderWeights


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DLLM_DEVICE']='cuda';os.environ['DLLM_ENCODER_GRAPH_MAX_BATCH']='64'
    torch.set_num_threads(4)
    model=EncoderWeights(args.model,512);executor=EncoderExecutor(model)
    docs=['Berlin had a population of 3,520,031 in 2015.','Berlin is well known for its museums.',
        'Paris is the capital of France.','A puppy is playing in a garden.']
    rows=[]
    def timed(tokens,types):
        torch.cuda.synchronize();start=time.perf_counter()
        result=executor.rerank(tokens,types);torch.cuda.synchronize()
        return result,(time.perf_counter()-start)*1000
    with torch.inference_mode():
        for count in (1,4,16,64):
            tokens,types=model.encode_pairs('How many people live in Berlin?',(docs*16)[:count])
            executor.use_graph=False
            expected,_=timed(tokens,types)
            eager=[timed(tokens,types)[1] for _ in range(10)]
            executor.use_graph=True
            actual,cold=timed(tokens,types)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            graph=[timed(tokens,types)[1] for _ in range(10)]
            saved=actual.clone()
            # Preserve the shape while changing the query/document boundary,
            # every token type, and token IDs; replay must upload all inputs.
            changed=[list(reversed(row)) for row in tokens]
            segments=[[1-t for t in row] for row in types]
            replay,_=timed(changed,segments)
            executor.use_graph=False;reference,_=timed(changed,segments);executor.use_graph=True
            torch.testing.assert_close(replay,reference,atol=0,rtol=0)
            torch.testing.assert_close(actual,saved,atol=0,rtol=0)
            assert len(executor.graphs)<=4
            rows.append({'batch':count,'length':max(map(len,tokens)),'cold_ms':cold,
                'eager_ms':eager,'graph_ms':graph,'median_eager_ms':statistics.median(eager),
                'median_graph_ms':statistics.median(graph),'exact_replay':True})
        # The padded-token bound excludes expensive large graph captures.
        keys=list(executor.graphs)
        tokens,types=model.encode_pairs('hello',['blue '*65]*64)
        timed(tokens,types);assert list(executor.graphs)==keys
        refs=[weakref.ref(entry[0]) for entry in executor.graphs.values()]
        for length in (3,7,15,31):
            tokens,types=model.encode_pairs('hello',['blue '*length]*16)
            timed(tokens,types)
        gc.collect();assert all(ref() is None for ref in refs)
        torch.cuda.synchronize();before=torch.cuda.memory_allocated()
        refs=[weakref.ref(entry[0]) for entry in executor.graphs.values()]
        executor.graphs.clear();gc.collect();torch.cuda.synchronize()
        after=torch.cuda.memory_allocated();assert all(ref() is None for ref in refs)
        assert after<before
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),
        'model':args.model,'precision':'float32','rows':rows,'padded_token_limit':4096,
        'large_batch_fallback':True,'eviction_reclaims_graphs':True,
        'memory_before_clear':before,'memory_after_clear':after},indent=2))


if __name__=='__main__':main()

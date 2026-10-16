"""Matched A2 FP32 cross-encoder HTTP comparison; real engines, same checkpoint."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen


PAIR_COUNTS=(1,4,16,64)
CONCURRENCIES=(1,4)
TRIALS=6


def _coverage(report):
    """Require every workload and measured trial emitted by this runner."""
    expected=[(count,concurrency) for count in PAIR_COUNTS for concurrency in CONCURRENCIES]
    rows=report.get('rows') if isinstance(report,dict) else None
    if not isinstance(rows,list) or len(rows)!=len(expected):
        raise ValueError('incomplete rerank workload coverage')
    for row,(count,concurrency) in zip(rows,expected):
        if (not isinstance(row,dict) or type(row.get('pairs_per_request')) is not int
                or type(row.get('concurrency')) is not int
                or (row['pairs_per_request'],row['concurrency'])!=(count,concurrency)):
            raise ValueError('invalid rerank workload grid')
        trials=row.get('trials')
        if not isinstance(trials,list) or len(trials)!=TRIALS:
            raise ValueError('incomplete rerank trial coverage')
        for index,trial in enumerate(trials):
            if not isinstance(trial,dict) or trial.get('warmup') is not (index==0):
                raise ValueError('invalid rerank warmup sequence')
            requests=trial.get('requests')
            if (not isinstance(requests,list) or len(requests)!=concurrency
                    or any(not isinstance(request,dict) or 'results' not in request for request in requests)):
                raise ValueError('incomplete rerank request coverage')


def _scores(results,count):
    """Validate complete per-document evidence before indexing or timing credit."""
    if type(count) is not int or count<1 or not isinstance(results,list) or len(results)!=count:
        raise ValueError('rerank results require exact document cardinality')
    scores={}
    for result in results:
        if not isinstance(result,dict):raise ValueError('invalid rerank result')
        index=result.get('index');score=result.get('relevance_score')
        if type(index) is not int or not 0<=index<count or index in scores:
            raise ValueError('invalid or duplicate rerank document index')
        if type(score) not in (int,float) or not -sys.float_info.max<=score<=sys.float_info.max:
            raise ValueError('rerank score must be finite numeric evidence')
        scores[index]=score
    return scores


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--engines',nargs='+',default=['dllm','vllm']);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    root=Path.cwd();reports=[]
    query='How many people live in Berlin?'
    base_docs=['Berlin had a population of 3,520,031 in 2015.','Berlin is well known for its museums.',
        'Paris is the capital of France.','A puppy is playing in a garden.']
    for engine in args.engines:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=dict(os.environ)
        if engine=='dllm':
            command=[sys.executable,'-m','dllm','serve','--port',str(port),'--context','512','--model','rank='+args.model]
        elif engine=='vllm':
            venv=root/'baselines/vllm-venv/bin'
            command=[str(venv/'vllm'),'serve',args.model,'--runner','pooling','--dtype','float',
                '--host','127.0.0.1','--port',str(port),'--served-model-name','rank','--max-model-len','512',
                '--max-num-seqs','64','--max-num-batched-tokens','8192','--gpu-memory-utilization','.5',
                '--no-enable-prefix-caching']
            env.update(PATH=str(venv)+':'+env.get('PATH',''),VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
                HF_HUB_OFFLINE='1',VLLM_CACHE_ROOT=str(root/'baselines/vllm-cache'),
                TORCHINDUCTOR_COMPILE_THREADS='2',MAX_JOBS='2',
                VLLM_FLOAT32_MATMUL_PRECISION='highest',TRITON_F32_DEFAULT='ieee',NVIDIA_TF32_OVERRIDE='0')
        else:raise ValueError('unknown engine')
        with open(f'rerank-compare-{engine}.log','w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=log,start_new_session=True)
            try:
                url=f'http://127.0.0.1:{port}'
                deadline=time.monotonic()+900
                while time.monotonic()<deadline:
                    if process.poll() is not None:raise RuntimeError(engine+' exited; inspect log')
                    try:
                        with urlopen(url+'/health',timeout=2) as response:
                            if response.status==200:break
                    except OSError:time.sleep(.5)
                else:raise TimeoutError('server startup timeout')
                rows=[]
                def request(documents):
                    body={'model':'rank','query':query,'documents':documents,'top_n':len(documents)}
                    # This checkpoint declares an identity output activation. Leave it
                    # enabled: vLLM 0.30 also gates the required BERT pooler tanh.
                    if engine=='vllm':body['use_activation']=True
                    else:body['return_documents']=True
                    start=time.perf_counter()
                    try:
                        with urlopen(Request(url+'/v1/rerank',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:
                            result=json.load(r)
                    except HTTPError as error:
                        error.close()
                        raise RuntimeError(f'HTTP {error.code}') from None
                    elapsed=time.perf_counter()-start
                    if not isinstance(result,dict):raise ValueError('invalid rerank response')
                    _scores(result.get('results'),len(documents))
                    return {'seconds':elapsed,'results':result['results'],'usage':result.get('usage')}
                for count in PAIR_COUNTS:
                    documents=(base_docs*16)[:count]
                    for concurrency in CONCURRENCIES:
                        trials=[]
                        with ThreadPoolExecutor(concurrency) as pool:
                            for index in range(TRIALS):
                                start=time.perf_counter();requests=list(pool.map(request,[documents]*concurrency))
                                elapsed=time.perf_counter()-start
                                trials.append({'warmup':index==0,'seconds':elapsed,'pairs_per_second':count*concurrency/elapsed,
                                    'requests':requests})
                        rows.append({'pairs_per_request':count,'concurrency':concurrency,'trials':trials,
                            'median_request_ms':1000*statistics.median(r['seconds'] for t in trials[1:] for r in t['requests']),
                            'median_pairs_per_second':statistics.median(t['pairs_per_second'] for t in trials[1:])})
                report={'engine':engine,'model':args.model,'command':command,'rows':rows,'precision':'float32',
                    'query':query,'base_documents':base_docs,'gpu_uuid':env['CUDA_VISIBLE_DEVICES'],
                    'vllm_use_activation':True if engine=='vllm' else None,
                    'precision_environment':{k:env.get(k) for k in ('VLLM_FLOAT32_MATMUL_PRECISION','TRITON_F32_DEFAULT','NVIDIA_TF32_OVERRIDE')},
                    'dllm_encoder_graph_max_batch':env.get('DLLM_ENCODER_GRAPH_MAX_BATCH','8')}
                Path(f'rerank-compare-{engine}.json').write_text(json.dumps(report,indent=2));reports.append(report)
                print(engine+' complete',flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    if len(reports)==2:
        Path('rerank-compare-agreement.json').write_text(json.dumps(compare_reports(*reports),indent=2))


def compare_reports(left,right):
    """Check every trial/request, including warmups, by original document index."""
    _coverage(left);_coverage(right)
    assert left['query']==right['query'] and left['base_documents']==right['base_documents']
    assert left['model']==right['model'] and left['precision']==right['precision']
    assert left['gpu_uuid']==right['gpu_uuid']
    assert len(left['rows'])==len(right['rows'])
    errors=[];checked=0
    for a,b in zip(left['rows'],right['rows']):
        assert (a['pairs_per_request'],a['concurrency'])==(b['pairs_per_request'],b['concurrency'])
        reference=a['trials'][-1]['requests'][0]
        expected=_scores(reference['results'],a['pairs_per_request'])
        maximum=0
        for row in (a,b):
            for trial in row['trials']:
                assert len(trial['requests'])==row['concurrency']
                for request in trial['requests']:
                    actual=_scores(request['results'],row['pairs_per_request'])
                    difference=max(abs(expected[i]-actual[i]) for i in expected)
                    assert difference<.0002,(a['pairs_per_request'],difference)
                    assert request['usage']==reference['usage']
                    maximum=max(maximum,difference);checked+=1
        errors.append(maximum)
    return {'passed':True,'max_errors':errors,'requests_checked':checked,'absolute_tolerance':.0002}


if __name__=='__main__':main()

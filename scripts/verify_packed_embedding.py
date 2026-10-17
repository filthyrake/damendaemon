"""Actual CUDA bitwise lookup, graph replay and complete-model storage parity."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
from damendaemon.device import empty_cache, device_name
from damendaemon.gguf_model import GgufWeights
from damendaemon.quantization import dequantize, dequantize_q4
from damendaemon.quantized_embedding import embedding
from damendaemon.runtime import require_gpu, make_executor


def expanded(model):
    result = copy.copy(model)
    weights, scales = dict(model.weights), dict(model.scales)
    unpack = dequantize_q4 if weights['embed'].dtype == torch.uint8 else dequantize
    shape = (model.config['vocab_size'], model.config['hidden_size'])
    dense = torch.empty(shape, dtype=torch.float16)
    for start in range(0, len(dense), 128):
        dense[start:start+128] = unpack(weights['embed'][start:start+128], scales['embed'][start:start+128], torch.float16)
    weights['embed'] = dense
    del scales['embed']
    result.finalize_weights(weights, scales)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--skip-timing',action='store_true'); args = p.parse_args()
    require_gpu(); torch.set_num_threads(4); torch.manual_seed(219)
    report = {'device':device_name(), 'torch':torch.__version__, 'kernel':[], 'model':[]}
    with torch.inference_mode():
        for q4 in (False, True):
            for width in (32, 96, 896, 1024, 4096):
                weight = torch.randint(0 if q4 else -128, 256 if q4 else 128,
                    (71, width//2 if q4 else width), dtype=torch.uint8 if q4 else torch.int8, device='cuda')
                scales = (torch.randn(71, width//32, device='cuda')*.013).half()
                scales[0] = 0
                dense = (dequantize_q4 if q4 else dequantize)(weight, scales, torch.float16)
                for dtype in (torch.float16, torch.float32):
                    for count in (1, 4, 129):
                        ids = (torch.arange(count*2, device='cuda').reshape(count,2)[:,0] % 71)
                        actual = embedding(ids, weight, scales, dtype)
                        assert torch.equal(actual, dense[ids].to(dtype)), (q4,width,dtype,count)
                        report['kernel'].append({'format':'q4_0' if q4 else 'q8_0','width':width,
                            'tokens':count,'dtype':str(dtype),'bit_exact':True})
                    staging = torch.tensor([0, 70, 1, 70], device='cuda')
                    for _ in range(3):embedding(staging, weight, scales, dtype)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): output = embedding(staging, weight, scales, dtype)
                    staging.copy_(torch.tensor([70,0,70,2],device='cuda'))
                    graph.replay()
                    assert torch.equal(output, dense[staging].to(dtype))
                    empty = embedding(staging[:0], weight, scales, dtype)
                    assert empty.shape==(0,width)
                    del graph, output
        report['graph_replay_changed_ids'] = True
        if args.model:
            digest=hashlib.sha256()
            with open(args.model,'rb') as file:
                for data in iter(lambda:file.read(1024*1024),b''):digest.update(data)
            report['sha256']=digest.hexdigest()
            model = GgufWeights(args.model, 512, 'none'); reference = expanded(model)
            report.update(path=str(Path(args.model).resolve()), packed_host_bytes=model.bytes,
                          expanded_host_bytes=reference.bytes, saved_host_bytes=reference.bytes-model.bytes)
            for paged in (False, True):
                if paged:os.environ['DAMENDAEMON_KV_CACHE_TOKENS']='1024'
                else:os.environ.pop('DAMENDAEMON_KV_CACHE_TOKENS',None)
                for graph in (False, True):
                    a=make_executor(reference,use_graph=graph,max_batch=4)
                    b=make_executor(model,use_graph=graph,max_batch=4)
                    for length in (1,17,129,505):
                        ids=([9707,11,1879,0]*127)[:length]
                        expected, actual = a.prefill(ids),b.prefill(ids)
                        for step in range(8):
                            assert torch.equal(expected,actual),(paged,graph,length,step,float((expected-actual).abs().max()))
                            token=expected.argmax(-1).item()
                            if step<7:expected,_=a.decode(token);actual,_=b.decode(token)
                        report['model'].append({'paged':paged,'graph':graph,'prompt_tokens':length,'exact_steps':8})
                    prompts=[[9707,11,1879,0]*size for size in (1,4,8,16)]
                    expected=a.prefill_batch(prompts,[0,1,2,3]);actual=b.prefill_batch(prompts,[0,1,2,3])
                    for _ in range(8):
                        assert torch.equal(expected,actual),('batch',paged,graph)
                        tokens=expected.argmax(-1).tolist()
                        expected,_=a.decode_batch(tokens,[0,1,2,3]);actual,_=b.decode_batch(tokens,[0,1,2,3])
                    for start in range(0,128,32):
                        expected=a.prefill_chunk([9707,11,1879,0]*8,0,start,final=start==96)
                        actual=b.prefill_chunk([9707,11,1879,0]*8,0,start,final=start==96)
                    assert torch.equal(expected,actual),('chunk',paged,graph)
                    del a,b,expected,actual
                    gc.collect();empty_cache()
            os.environ.pop('DAMENDAEMON_KV_CACHE_TOKENS',None)
            report['batch_and_chunk_exact'] = True
            del model,reference
            gc.collect();empty_cache()
            if not args.skip_timing:
                timings=[]
                # CPU-expanded mode reproduces the previous Q8 loader. It is used
                # only during loading; both executors and all timings use CUDA.
                for turn in range(4):
                    for mode in (('packed','expanded') if turn%2==0 else ('expanded','packed')):
                        started=time.perf_counter()
                        with patch('damendaemon.gguf_model.selected_device',return_value='cuda' if mode=='packed' else 'mps'):
                            model=GgufWeights(args.model,512,'none')
                        loaded=time.perf_counter()
                        before=torch.cuda.memory_allocated()
                        executor=make_executor(model,use_graph=True,max_batch=4)
                        torch.cuda.synchronize();ready=time.perf_counter()
                        allocated=torch.cuda.memory_allocated()-before
                        samples=[]
                        for _ in range(9):
                            start=time.perf_counter();logits=executor.prefill([9707,11,1879,0]*32)
                            tokens=[]
                            for step in range(16):
                                token=logits.argmax(-1).item();tokens.append(token)
                                if step<15:logits,_=executor.decode(token)
                            torch.cuda.synchronize();samples.append(time.perf_counter()-start)
                        timings.append(dict(round=turn,mode=mode,host_load_seconds=loaded-started,
                            activation_seconds=ready-started,gpu_allocated_bytes=allocated,
                            warm_generation_seconds=samples[2:],output=tokens))
                        del executor,model,logits
                        gc.collect();empty_cache()
                assert len({tuple(t['output']) for t in timings})==1
                report['timings']=timings
                report['timing_medians']={mode:{key:statistics.median(t[key] for t in timings if t['mode']==mode and t['round']>0)
                    for key in ('host_load_seconds','activation_seconds','gpu_allocated_bytes')} for mode in ('packed','expanded')}
                for mode in report['timing_medians']:
                    report['timing_medians'][mode]['warm_generation_seconds']=statistics.median(
                        sample for t in timings if t['mode']==mode for sample in t['warm_generation_seconds'])
    report['passed']=True
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

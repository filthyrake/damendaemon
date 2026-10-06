"""Real prefill grouping, numerical, latency and allocation probe on the A2.

This explicitly overrides an executor's aggregate budget; normal defaults stay
unchanged. Timings cover synchronized prefill calls, not serving or decode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from dllm.gguf_model import GgufWeights
from dllm.runtime import make_executor, require_gpu


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require_gpu()
    if torch.cuda.get_device_name()!='NVIDIA A2' or os.environ.get('DLLM_KV_CACHE_TOKENS'):
        raise ValueError('probe requires an A2 with ordinary fixed KV slots')
    torch.set_num_threads(4)
    source=GgufWeights(args.model,512,'none')
    library=Path(os.environ['DLLM_Q4_MARLIN_LIBRARY'])
    report=dict(passed=False,model=args.model,device=torch.cuda.get_device_name(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        context_per_slot=512,concurrency=4,cases=[],
        scope='synchronized prefill only; explicit aggregate-budget override; unchanged production admission')
    with Path(args.model).open('rb') as stream:
        report['model_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    references={}
    try:
        executor=make_executor(source,use_graph=True,max_batch=4)
        if executor.q4_marlin_abi!=2 or len(executor.quant_linear.q4_marlin)!=129:
            raise ValueError('probe requires qualified Q4 Marlin ABI 2 projections')
        report['admitted_required_bytes']=executor.required_memory_bytes
        report['prefill_graphs_enabled']=executor.prefill_graph_cache is not None
        corpus=source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
        for phase,budget in enumerate((512,1024,2048,2048,1024,512)):
            executor.prefill_token_budget=budget
            for length in (129,400,512):
                ids=(corpus*((length+len(corpus)-1)//len(corpus)))[:length]
                prompts=[ids]*4
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                shapes=[]
                originals={name:getattr(executor,name) for name in ('forward_hidden','forward_packed_hidden')}
                def traced(name,forward):
                    def call(x,*values,**options):
                        shapes.append(dict(path=name,rows=int(x.shape[0])))
                        return forward(x,*values,**options)
                    return call
                try:
                    for name,forward in originals.items():setattr(executor,name,traced(name,forward))
                    logits=executor.prefill_batch(prompts,list(range(4)))
                finally:
                    for name in originals:delattr(executor,name)
                actual=logits.cpu().float()
                if length not in references:references[length]=actual.clone()
                expected=references[length]
                error=actual-expected
                rms=(error.square().mean()/expected.square().mean().clamp_min(1e-12)).sqrt().item()
                maximum=error.abs().max().item()
                correct=bool(torch.isfinite(actual).all()) and rms<.01 and maximum<.5 and torch.equal(actual.argmax(-1),expected.argmax(-1))
                case=dict(phase=phase,budget=budget,prompt_tokens=length,observed_forwards=shapes,
                          relative_rms=rms,max_absolute=maximum,correct=correct,seconds=[])
                report['cases'].append(case)
                if correct:
                    for _ in range(2):logits=executor.prefill_batch(prompts,list(range(4)))
                    for _ in range(5):
                        torch.cuda.synchronize()
                        started=time.perf_counter()
                        logits=executor.prefill_batch(prompts,list(range(4)))
                        torch.cuda.synchronize()
                        case['seconds'].append(time.perf_counter()-started)
                    case['median_seconds']=statistics.median(case['seconds'])
                case['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                case['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                case['allocated_peak_within_admission']=case['peak_allocated_bytes']<=executor.required_memory_bytes
                print(phase,budget,length,case.get('median_seconds'),case['allocated_peak_within_admission'],flush=True)
                Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        report['passed']=all(c['correct'] and c['allocated_peak_within_admission'] for c in report['cases'])
        if not report['passed']:raise AssertionError('prefill numerical or allocation gate failed')
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

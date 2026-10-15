"""CPU-only production grammar setup cost, excluding weights and inference."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from dllm.structured import Constraint, prepare_options, response_schema


def schema(value):
    return response_schema(dict(type='json_schema', json_schema=dict(name='bounded', schema=dict(
        type='object', properties={'answer':dict(type='integer',const=value)},
        required=['answer'],additionalProperties=False))))


def benchmark(model_factory, trials=3, repeats=5, constraint=Constraint, clock=time.perf_counter):
    if type(trials) is not int or not 1<=trials<=5 or type(repeats) is not int or not 1<=repeats<=10:
        raise ValueError('trials1..5 and repeats1..10 required')
    samples=[]
    def measure(phase,trial,iteration,action):
        start=clock(); result=action(); elapsed=clock()-start
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid CPU timing')
        samples.append(dict(phase=phase,trial=trial,iteration=iteration,seconds=elapsed))
        return result
    for trial in range(trials):
        model=model_factory()
        initial=schema(0)
        measure('cold_constraint',trial,0,lambda:constraint(model,initial))
        for iteration in range(repeats):
            measure('warm_constraint',trial,iteration,lambda:constraint(model,initial))
            changed=schema(iteration+1)
            measure('changed_schema',trial,iteration,lambda:constraint(model,changed))
            measure('unconstrained_options',trial,iteration,lambda:prepare_options(None,[],None))
        del model
    phases=sorted({r['phase'] for r in samples})
    return dict(samples=samples,medians_seconds={p:statistics.median(r['seconds'] for r in samples if r['phase']==p) for p in phases},
                trials=trials,repeats=repeats,device='cpu',model_weights_loaded=False,inference=False,
                gpu_performance_qualified=False,
                scope='Constraint constructor including matcher/mask; cold includes compiler/tokenizer setup; not isolated compile kernel')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':parser.error('requires CUDA_VISIBLE_DEVICES empty')
    if args.output.exists():parser.error('new output path required')
    import torch
    from tokenizers import Tokenizer
    torch.set_num_threads(1)
    tokenizer_path=args.snapshot/'tokenizer.json'
    data=tokenizer_path.read_bytes()
    config=json.loads((args.snapshot/'config.json').read_text())
    generation=json.loads((args.snapshot/'generation_config.json').read_text())
    eos=generation.get('eos_token_id',config.get('eos_token_id'))
    eos=[eos] if type(eos) is int else eos
    if not eos or any(type(t) is not int for t in eos):raise ValueError('valid EOS IDs required')
    def model():
        return SimpleNamespace(tokenizer=Tokenizer.from_str(data.decode()),config=config,eos=eos)
    report=benchmark(model)
    if tokenizer_path.read_bytes()!=data:raise ValueError('tokenizer changed')
    report.update(tokenizer_sha256=hashlib.sha256(data).hexdigest(),
                  snapshot=str(args.snapshot),vocab_size=config['vocab_size'],
                  versions={p:version(p) for p in ('torch','xgrammar','transformers','tokenizers')})
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

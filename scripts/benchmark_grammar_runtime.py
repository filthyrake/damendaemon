"""Paired grammar-component cost; no model inference or end-to-end speed claim."""
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

from damendaemon.structured import Constraint
from scripts.benchmark_grammar_setup import schema


def measure(model, tokens, pairs=10, device='cpu', constraint=Constraint, clock=time.perf_counter):
    import torch
    if type(pairs) is not int or not 2<=pairs<=20 or not 1<=len(tokens)<=32:
        raise ValueError('pairs2..20 and tokens1..32 required')
    vocab=model.config['vocab_size']
    if any(type(t) is not int or not 0<=t<vocab for t in tokens):raise ValueError('invalid scripted token')
    sync=torch.cuda.synchronize if device=='cuda' else lambda:None
    logits=torch.zeros(vocab,device=device,dtype=torch.float32)
    raw=[]
    def arm(constrained,pair,record=True):
        matcher=constraint(model,schema(0)) if constrained else None
        for step,token in enumerate(tokens):
            logits.zero_();logits[token]=1
            original=logits.clone()
            sync();start=clock()
            result=matcher.apply(logits) if matcher else logits
            if matcher:matcher.accept(token)
            sync();elapsed=clock()-start
            if not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid timing')
            if not torch.equal(logits,original):raise ValueError('constraint mutated source logits')
            if not torch.isfinite(result[token]) or int(result.argmax())!=token:
                raise ValueError('scripted token not permitted by grammar')
            if record:raw.append(dict(pair=pair,arm='constrained' if constrained else 'plain',
                                     step=step,token=token,seconds=elapsed))
    # Warm compiler, matcher and mask kernel before any measured arm.
    arm(True,-1,False);arm(False,-1,False)
    for pair in range(pairs):
        for constrained in ((False,True) if pair%2==0 else (True,False)):
            arm(constrained,pair)
    deltas=[]
    for pair in range(pairs):
        totals={name:sum(r['seconds'] for r in raw if r['pair']==pair and r['arm']==name) for name in ('plain','constrained')}
        deltas.append((totals['constrained']-totals['plain'])/len(tokens))
    return dict(samples=raw,paired_extra_seconds_per_token=deltas,
                median_extra_seconds_per_token=statistics.median(deltas),
                medians_seconds={name:statistics.median(r['seconds'] for r in raw if r['arm']==name) for name in ('plain','constrained')},
                tokens=tokens,pairs=pairs,device=device,vocab_size=vocab,
                scope='warm Constraint.apply and accept; matched FP32 logits; compilation and inference excluded',
                end_to_end_performance_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.output.exists():parser.error('new report path required')
    if args.device=='cuda':
        uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('DAMENDAEMON_SHARED_CUDA_GUARD')!='1':
            parser.error('CUDA requires shared guard and explicit UUID')
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':parser.error('CPU requires CUDA_VISIBLE_DEVICES empty')
    import torch
    from tokenizers import Tokenizer
    torch.set_num_threads(1)
    if args.device=='cuda':
        if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
        torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    data=(args.snapshot/'tokenizer.json').read_bytes()
    tokenizer=Tokenizer.from_str(data.decode())
    config=json.loads((args.snapshot/'config.json').read_text())
    generation=json.loads((args.snapshot/'generation_config.json').read_text())
    eos=generation.get('eos_token_id',config.get('eos_token_id'))
    eos=[eos] if type(eos) is int else eos
    if not eos or any(type(t) is not int for t in eos):raise ValueError('invalid EOS metadata')
    model=SimpleNamespace(tokenizer=tokenizer,config=config,eos=eos)
    tokens=tokenizer.encode('{"answer":0}',add_special_tokens=False).ids+[eos[0]]
    report=measure(model,tokens,device=args.device)
    report.update(tokenizer_sha256=hashlib.sha256(data).hexdigest(),
                  versions={p:version(p) for p in ('torch','xgrammar','transformers','tokenizers')})
    if args.device=='cuda':report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

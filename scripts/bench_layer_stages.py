"""Matched A2/A30 sequential stage graph comparison; no cross-host claims."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from dllm.qwen2 import ModelWeights
from scripts.experiments.layer_stage import LayerStage


def sync():
    for device in range(2):torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(model,prompt,graphs,repeats,output_tokens):
    stages=[]
    total=model.config['num_hidden_layers'];split=total//2
    for device,(start,end) in enumerate(((0,split),(split,total))):
        with torch.cuda.device(device):
            stages.append(LayerStage(model,start,end,max_batch=1,stage_graphs=graphs))
    results=[]
    for iteration in range(repeats+2):
        sync();started=time.perf_counter()
        hidden=None
        for device,stage in enumerate(stages):
            with torch.cuda.device(device):
                inputs=torch.tensor(prompt,device='cuda') if device==0 else hidden.to(device)
                hidden=stage.run_prefill(inputs,torch.arange(len(prompt),device='cuda'),0)
        token=hidden.argmax(-1).item();generated=[token]
        sync();first=time.perf_counter()
        for position in range(len(prompt),len(prompt)+output_tokens-1):
            for device,stage in enumerate(stages):
                with torch.cuda.device(device):
                    inputs=torch.tensor([token],device='cuda') if device==0 else hidden.to(device)
                    hidden=stage.run_decode(inputs,torch.tensor([position],device='cuda'),torch.tensor([0],device='cuda'))
            token=hidden.argmax(-1).item();generated.append(token)
        sync();finished=time.perf_counter()
        if iteration>=2:
            results.append(dict(first_token_seconds=first-started,total_seconds=finished-started,
                decode_tokens_per_second=(output_tokens-1)/(finished-first),tokens=generated))
    return dict(stage_graphs=graphs,runs=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=6);p.add_argument('--output-tokens',type=int,default=32)
    args=p.parse_args()
    if torch.cuda.device_count()!=2:p.error('select exactly two GPUs')
    if args.repeats<1 or args.output_tokens<2:p.error('invalid repetitions or output length')
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    prompt=next(prompt for event in trace['trace'] if event['kind']=='prefill' for prompt in event['prompts'] if len(prompt)==255)
    if len(prompt)+args.output_tokens>trace['context']:p.error('output exceeds context')
    model=ModelWeights(trace['path'],trace['context'],quantization=trace.get('reference_quantization','none'))
    groups=[]
    for graphs in (False,True,True,False):
        groups.append(measure(model,prompt,graphs,args.repeats,args.output_tokens))
        gc.collect()
        for device in range(2):
            with torch.cuda.device(device):torch.cuda.empty_cache()
    expected=groups[0]['runs'][0]['tokens']
    passed=all(row['tokens']==expected for group in groups for row in group['runs'])
    summaries=[]
    for graphs in (False,True):
        rows=[row for group in groups if group['stage_graphs']==graphs for row in group['runs']]
        summaries.append(dict(stage_graphs=graphs,**{key:statistics.median(r[key] for r in rows)
            for key in ('first_token_seconds','total_seconds','decode_tokens_per_second')}))
    report=dict(passed=passed,model=trace['path'],quantization=model.quantization,context=trace['context'],
        prompt_tokens=len(prompt),output_tokens=args.output_tokens,devices=[torch.cuda.get_device_name(i) for i in range(2)],
        torch=str(torch.__version__),scope='same-host sequential stage graph ABBA; no network or serving throughput claim',
        summaries=summaries,groups=groups)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries))
    if not passed:raise AssertionError('greedy outputs changed between graph modes')


if __name__=='__main__':main()

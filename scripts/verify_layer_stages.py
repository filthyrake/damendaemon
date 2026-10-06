"""Experimental two-device contiguous layers against a trusted reference trace."""
import argparse
import json
import time
import torch
from dllm.qwen2 import ModelWeights
from scripts.experiments.layer_stage import LayerStage


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage-graphs',action='store_true')
    args=parser.parse_args()
    if torch.cuda.device_count()!=2:parser.error('select exactly two CUDA GPUs')
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    model=ModelWeights(trace['path'],trace['context'],quantization=trace.get('reference_quantization','none'))
    total=model.config['num_hidden_layers']; split=total//2
    stages=[]; residency=[]
    for device,(start,end) in enumerate(((0,split),(split,total))):
        with torch.cuda.device(device):
            stage=LayerStage(model,start,end,max_batch=4,stage_graphs=args.stage_graphs)
            stages.append(stage)
            assert len(stage.kv)==end-start
            assert ('embed' in stage.w)==(device==0)
            assert ('head' in stage.w)==(device==1)
            assert all(t.device.index==device for t in stage.w.values())
            scales=list(stage.quant_linear.scales.values()) if stage.quant_linear else []
            assert all(t.device.index==device for t in scales)
            residency.append(dict(device=torch.cuda.get_device_name(device),layers=[start,end],
                weights_bytes=sum(t.numel()*t.element_size() for t in {id(t):t for t in stage.w.values()}.values()),
                scale_bytes=sum(t.numel()*t.element_size() for t in {id(t):t for t in scales}.values()),
                kv_bytes=sum(t.numel()*t.element_size() for t in stage.kv)))
    checks=[]; timings=[]; lengths=[0]*4
    for event in trace['trace']:
        for device in range(2):torch.cuda.synchronize(device)
        started=time.perf_counter()
        if event['kind']=='prefill':
            outputs=[]
            for prompt,slot in zip(event['prompts'],event['slots']):
                hidden=None
                for device,stage in enumerate(stages):
                    with torch.cuda.device(device):
                        inputs=torch.tensor(prompt,device='cuda') if device==0 else hidden.to(device)
                        hidden=stage.run_prefill(inputs,torch.arange(len(prompt),device='cuda'),slot)
                outputs.append(hidden.cpu());lengths[slot]=len(prompt)
            actual=torch.cat(outputs)
        else:
            hidden=None
            for device,stage in enumerate(stages):
                with torch.cuda.device(device):
                    inputs=torch.tensor(event['tokens'],device='cuda') if device==0 else hidden.to(device)
                    positions=torch.tensor([lengths[s] for s in event['slots']],device='cuda')
                    slots=torch.tensor(event['slots'],device='cuda')
                    hidden=stage.run_decode(inputs,positions,slots)
            actual=hidden.cpu()
            for slot in event['slots']:lengths[slot]+=1
        for device in range(2):torch.cuda.synchronize(device)
        timings.append(dict(kind=event['kind'],seconds=time.perf_counter()-started))
        for row,target in enumerate(event['targets']):
            value=actual[row].float();target=target.squeeze(0)
            error=value-target
            relative=(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-6)).item()
            absolute=error.abs().max().item();match=value.argmax().item()==target.argmax().item()
            checks.append(dict(kind=event['kind'],slot=event['slots'][row],relative_rms=relative,
                max_absolute=absolute,greedy_match=match,passed=relative<.01 and absolute<.5 and match))
    report=dict(passed=bool(checks) and all(c['passed'] for c in checks),model=trace['path'],
        context=trace['context'],torch=str(torch.__version__),reference_torch=trace['reference_torch'],
        reference_dtype=trace['reference_dtype'],quantization=model.quantization,stage_graphs=args.stage_graphs,residency=residency,checks=checks,timings=timings,
        scope='experimental sequential layer partition; no serving, capacity, or speedup claim')
    from pathlib import Path
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),residency=residency)))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()

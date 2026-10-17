"""Isolate output-head arithmetic on the same native hidden states.

Diagnostic only: temporary dense head copies are never used by serving or fed
back into KV state. The unchanged native result advances the teacher-forced
trace. This is not a speed benchmark or an alternative qualification tolerance.
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
import os
from pathlib import Path


def metrics(value,target):
    error=value-target
    relative=(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1.e-6)).item()
    absolute=error.abs().max().item()
    match=value.argmax().item()==target.argmax().item()
    return dict(relative_rms=relative,max_absolute=absolute,greedy_match=match,
                passed=relative<.01 and absolute<.5 and match)


def main():
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True)
    body=p.add_mutually_exclusive_group()
    body.add_argument('--dense-body',action='store_true',help='diagnostic control: expand body projections to FP16')
    body.add_argument('--single-dot-body',action='store_true',help='diagnostic control: contiguous K-order packed products')
    p.add_argument('--torch-norm',action='store_true',help='diagnostic control: unfused FP16 RMS normalization')
    p.add_argument('--packed',action='store_true');args=p.parse_args()
    os.environ.update(DAMENDAEMON_DEVICE='cuda',DAMENDAEMON_QUANTIZATION='q4_0',DAMENDAEMON_Q4_STAGE_ADJACENT='0',
                      DAMENDAEMON_PARTITION_PACKED_PREFILL=str(int(args.packed)))
    import torch
    from damendaemon.qwen2 import ModelWeights
    from damendaemon.partition import PartitionExecutor
    from damendaemon.quantized_kernels_q4 import expand
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    if trace.get('reference_quantization')!='q4_0':raise ValueError('Q4 reference required')
    events=trace['trace'];capacity=1+max(max(event['slots']) for event in events)
    report=dict(scope='Eager head variants on identical hidden state per event; optional body arithmetic control; no performance or serving qualification claim',
        trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
        reference_dtype=trace['reference_dtype'],packed=args.packed,dense_body=args.dense_body,
        single_dot_body=args.single_dot_body,torch_norm=args.torch_norm,tf32=False,checks=[],
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ('damendaemon/quantized_kernels_q4.py','damendaemon/qwen2.py','damendaemon/layer_stage.py','damendaemon/partition.py',
                  'scripts/experiments/q4_prefill.py','scripts/experiments/diagnose_q4_head.py')})
    with torch.inference_mode():
        executor=PartitionExecutor(ModelWeights(trace['path'],trace['context']),12,use_graph=False,max_batch=capacity)
        stage=executor.stages[-1];original=stage.output_logits;observed=[]
        try:
            if args.torch_norm:
                for part in executor.stages:
                    def norm(x,weight,eps=part.eps):
                        xf=x.float()
                        return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)).to(weight.dtype)*weight
                    part.norm=norm
            if args.dense_body:
                for device,part in enumerate(executor.stages):
                    with torch.cuda.device(device):
                        expanded={id(weight):expand(weight,part.quant_linear.scales[id(weight)],torch.float16)
                                  for name,weight in part.w.items() if weight.dtype==torch.uint8 and name!='head'}
                    fallback=part.linear
                    def dense_body(x,weight,bias=None,expanded=expanded,fallback=fallback):
                        if id(weight) in expanded:return torch.nn.functional.linear(x,expanded[id(weight)],bias)
                        return fallback(x,weight,bias)
                    part.linear=dense_body
            if args.single_dot_body:
                import triton as tr
                from scripts.experiments.q4_prefill import single_dot
                for part in executor.stages:
                    fallback=part.linear;scales=part.quant_linear.scales;head=part.w.get('head')
                    def packed_body(x,weight,bias=None,fallback=fallback,scales=scales,head=head):
                        rows=x.numel()//x.shape[-1]
                        if weight is head or weight.dtype!=torch.uint8 or x.dtype!=torch.float16 or rows>128:
                            return fallback(x,weight,bias)
                        shape=x.shape[:-1];x=x.reshape(rows,-1).contiguous();k=x.shape[1];n=weight.shape[0]
                        output=torch.empty((rows,n),device=x.device,dtype=x.dtype)
                        bm=max(16,tr.next_power_of_2(rows))
                        single_dot[(tr.cdiv(rows,bm),tr.cdiv(n,64))](x,weight,scales[id(weight)],
                            weight if bias is None else bias,output,rows,n,k,bm,64,128,bias is not None,
                            num_warps=8 if bm==128 else 4,num_stages=3)
                        return output.reshape(*shape,n)
                    part.linear=packed_body
            with torch.cuda.device(executor.output_device):
                weight=stage.w['head']
                dense=expand(weight,stage.quant_linear.scales[id(weight)],torch.float16)
                dense_float=dense.float()
            def capture(hidden):
                native=original(hidden)
                normalized=stage.norm(hidden,stage.w['norm'])
                fp16=torch.nn.functional.linear(normalized,dense)
                fp32=torch.nn.functional.linear(normalized.float(),dense_float)
                observed.append((fp16.float().cpu(),fp32.cpu()))
                return native
            stage.output_logits=capture
            for index,event in enumerate(events):
                observed.clear()
                if event['kind']=='prefill':actual=executor.prefill_batch(event['prompts'],event['slots'])
                else:actual,_=executor.decode_batch(event['tokens'],event['slots'])
                native=actual.float().cpu()
                alternatives=[torch.cat([item[column] for item in observed]) for column in (0,1)]
                assert all(value.shape==native.shape for value in alternatives)
                for row,target in enumerate(event['targets']):
                    target=target.squeeze(0).float()
                    values=dict(native=native[row],dense_fp16=alternatives[0][row],dense_fp32=alternatives[1][row])
                    check=dict(event_index=index,kind=event['kind'],slot=event['slots'][row],
                               variants={key:metrics(value,target) for key,value in values.items()})
                    if any(not result['passed'] for result in check['variants'].values()):
                        check['top5']={key:list(zip(value.topk(5).indices.tolist(),value.topk(5).values.tolist()))
                                       for key,value in {**values,'reference':target}.items()}
                    report['checks'].append(check)
            report['summary']={key:dict(checks=len(report['checks']),
                failures=sum(not check['variants'][key]['passed'] for check in report['checks']))
                for key in ('native','dense_fp16','dense_fp32')}
            report['completed']=True
        finally:
            stage.output_logits=original
            executor.close()
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()

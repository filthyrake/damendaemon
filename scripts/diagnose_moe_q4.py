"""Locate Q4 MoE divergence with exact weight checks and observed expert routes.

Run through experiments/run_moe_q4.py. This diagnostic adds synchronization and
CPU copies; its timings are not performance measurements.
"""
import argparse
import json
from pathlib import Path
from types import MethodType
import torch
from reference_model import load_reference
from dllm.qwen2 import ModelWeights
from dllm.runtime import make_executor, require_gpu
from dllm.quantization import dequantize_q4
from dllm import expert_routes, moe_kernels


def metrics(value, target):
    value, target = value.float().cpu(), target.float().cpu()
    error = value-target
    return dict(relative_rms=(error.square().mean()/target.square().mean().clamp_min(1e-12)).sqrt().item(),
                max_absolute=error.abs().max().item())


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('trace',help='trusted local independent trace')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    require_gpu();torch.set_num_threads(4)
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    source=ModelWeights(trace['path'],trace['context'])
    executor=make_executor(source,use_graph=False,max_batch=4)
    reference=load_reference(trace['path'],reference_quantization='q4_0',local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager').eval()
    report=dict(completed=False,scope='diagnostic only; instrumented execution is not a benchmark',
                model=trace['path'],weight_checks=0,weight_failures=[],phases=[])
    state=reference.state_dict()
    def weight_check(native_key,reference_key,first=None,last=None):
        weight=source.weights[native_key]
        scales=source.scales.get(native_key)
        if first is not None:
            weight=weight[first:last]
            scales=None if scales is None else scales[first:last]
        actual=dequantize_q4(weight,scales) if weight.dtype==torch.uint8 else weight.float()
        expected=state[reference_key].float()
        if actual.shape!=expected.shape:raise AssertionError((native_key,reference_key,actual.shape,expected.shape))
        report['weight_checks']+=1
        if not torch.equal(actual,expected):
            report['weight_failures'].append(dict(native=native_key,reference=reference_key,**metrics(actual,expected)))
    try:
        weight_check('embed','model.embed_tokens.weight')
        weight_check('norm','model.norm.weight')
        weight_check('head','lm_head.weight')
        c=source.config;h=c['hidden_size'];width=c['intermediate_size']
        for layer in range(c['num_hidden_layers']):
            root=f'model.layers.{layer}.'
            for index,name in enumerate(('q','k','v')):
                weight_check(f'{layer}.qkv.weight',root+f'self_attn.{name}_proj.weight',index*h,(index+1)*h)
            for native,name in [('out','self_attn.o_proj'),('q_norm','self_attn.q_norm'),
                                ('k_norm','self_attn.k_norm'),('norm1','input_layernorm'),
                                ('norm2','post_attention_layernorm'),('router','mlp.gate')]:
                weight_check(f'{layer}.{native}',root+name+'.weight')
            for expert in range(c['num_experts']):
                base=expert*2*width
                weight_check(f'{layer}.experts_up',root+f'mlp.experts.{expert}.gate_proj.weight',base,base+width)
                weight_check(f'{layer}.experts_up',root+f'mlp.experts.{expert}.up_proj.weight',base+width,base+2*width)
                weight_check(f'{layer}.experts_down',root+f'mlp.experts.{expert}.down_proj.weight',expert*h,(expert+1)*h)
            print(f'weight equivalence layer={layer}',flush=True)
        if report['weight_failures']:
            raise AssertionError('runtime and independent reference weights differ')
        native_records,reference_records,current={}, {}, {}
        router_ids={id(executor.w[f'{i}.router']):i for i in range(c['num_hidden_layers'])}
        original_linear=executor._linear
        def tracked_linear(self,x,weight,bias=None):
            value=original_linear(x,weight,bias)
            layer=router_ids.get(id(weight))
            if layer is not None:
                current['layer']=layer
                native_records[layer]=dict(hidden=x.detach().cpu(),logits=value.detach().cpu())
            return value
        executor._linear=MethodType(tracked_linear,executor)
        original_groups=expert_routes.grouped_routes
        def tracked_groups(selected,experts):
            native_records[current['layer']]['selected']=selected.detach().cpu()
            return original_groups(selected,experts)
        expert_routes.grouped_routes=tracked_groups
        original_routed=moe_kernels.routed_linear
        def tracked_routed(x,weight,scales,selected,outputs,shared):
            native_records[current['layer']]['selected']=selected.detach().cpu()
            return original_routed(x,weight,scales,selected,outputs,shared)
        moe_kernels.routed_linear=tracked_routed
        def hook(layer):
            def record(module,inputs,output):
                reference_records[layer]=dict(hidden=inputs[0].detach().clone(),logits=output.detach().clone(),
                    selected=output.float().softmax(-1).topk(c['num_experts_per_tok'],dim=-1).indices)
            return record
        for layer,block in enumerate(reference.model.layers):
            block.mlp.gate.register_forward_hook(hook(layer))
        def phase(name,actual,target,oracle,row=None):
            result=dict(name=name,native_vs_reference=metrics(actual,target),
                        current_reference_vs_saved=metrics(target,oracle),layers=[])
            for layer in range(c['num_hidden_layers']):
                native=native_records[layer];ref=reference_records[layer]
                if row is not None:native={key:value[row:row+1] for key,value in native.items()}
                different=(native['selected'].sort(-1).values!=ref['selected'].sort(-1).values).any(-1)
                indices=different.nonzero().flatten().tolist()
                record=dict(layer=layer,hidden=metrics(native['hidden'],ref['hidden']),
                            router_logits=metrics(native['logits'],ref['logits']),different_route_rows=len(indices),examples=[])
                for index in indices[:4]:
                    example=dict(row=index,native_selected=native['selected'][index].tolist(),
                                 reference_selected=ref['selected'][index].tolist())
                    router=source.weights[f'{layer}.router'].double()
                    for label,values in (('native',native),('reference',ref)):
                        probabilities=values['logits'][index].float().softmax(-1)
                        top=probabilities.topk(c['num_experts_per_tok']+1)
                        precise=torch.nn.functional.linear(values['hidden'][index].double(),router).topk(c['num_experts_per_tok']+1)
                        example[label+'_top9']=dict(indices=top.indices.tolist(),probabilities=top.values.tolist(),
                            cutoff_gap=(top.values[-2]-top.values[-1]).item())
                        example[label+'_input_fp64_top9']=dict(indices=precise.indices.tolist(),
                            logits=precise.values.tolist(),cutoff_gap=(precise.values[-2]-precise.values[-1]).item())
                    record['examples'].append(example)
                result['layers'].append(record)
            report['phases'].append(result)
            print(json.dumps(result),flush=True)
        chosen=next((i,e) for i,e in enumerate(trace['trace'])
                    if e['kind']=='prefill' and any(len(prompt)==496 for prompt in e['prompts']))
        index,event=chosen
        row=next(i for i,prompt in enumerate(event['prompts']) if len(prompt)==496)
        slot=event['slots'][row]
        actual=executor.prefill_batch(event['prompts'],event['slots'])
        target=reference(torch.tensor([event['prompts'][row]]),use_cache=True)
        phase('prefill:496',actual[row],target.logits[:,-1],event['targets'][row])
        cache=target.past_key_values
        steps=0
        for event in trace['trace'][index+1:]:
            if event['kind']!='decode' or slot not in event['slots']:break
            row=event['slots'].index(slot)
            native_records.clear();reference_records.clear()
            actual,_=executor.decode_batch(event['tokens'],event['slots'])
            target=reference(torch.tensor([[event['tokens'][row]]]),past_key_values=cache,use_cache=True)
            cache=target.past_key_values
            phase(f'decode:{497+steps}',actual[row],target.logits[:,-1],event['targets'][row],row=row)
            steps+=1
            if steps==3:break
        report['completed']=True
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

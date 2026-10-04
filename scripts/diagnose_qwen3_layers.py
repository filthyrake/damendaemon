"""Layerwise Qwen3 residual drift on a saved independent teacher-forced trace."""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM
from dllm.qwen2 import ModelWeights
from dllm.batch import BatchExecutor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('trace')
    args=p.parse_args()
    torch.set_num_threads(4)
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    model=ModelWeights(trace['path'],trace['context'])
    executor=BatchExecutor(model,use_graph=False,max_batch=4)
    reference=AutoModelForCausalLM.from_pretrained(trace['path'],local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    actual_layers,target_layers={},{}
    norm=executor.norm
    names={id(executor.w[f'{i}.norm1']):i for i in range(executor.c['num_hidden_layers'])}
    def capture_norm(x,weight):
        if id(weight) in names:actual_layers[names[id(weight)]]=x.detach().float().cpu()
        return norm(x,weight)
    executor.norm=capture_norm
    handles=[]
    for i,layer in enumerate(reference.model.layers):
        def hook(module,args,layer_index=i):target_layers[layer_index]=args[0].detach().float().cpu()
        handles.append(layer.register_forward_pre_hook(hook))
    states,checks={},[]
    def metric(a,b):
        a,b=a.float(),b.float();e=a-b
        return {'relative_rms':(e.square().mean()/b.square().mean().clamp_min(1e-12)).sqrt().item(),
                'max_absolute':e.abs().max().item()}
    with torch.inference_mode():
        for event in trace['trace']:
            if event['kind']=='prefill':
                executor.prefill_batch(event['prompts'],event['slots'])
                for slot,ids in zip(event['slots'],event['prompts']):
                    r=reference(torch.tensor([ids],device='cuda'),use_cache=True,logits_to_keep=1)
                    states[slot]=r.past_key_values
                continue
            logits,_=executor.decode_batch(event['tokens'],event['slots'])
            for row,(slot,token) in enumerate(zip(event['slots'],event['tokens'])):
                r=reference(torch.tensor([[token]],device='cuda'),past_key_values=states[slot],use_cache=True,logits_to_keep=1)
                states[slot]=r.past_key_values
                error=metric(logits[row].cpu(),r.logits[0,-1].cpu())
                if error['relative_rms']>.005:
                    checks.append({'slot':slot,'length':executor.lengths[slot],'logits':error,
                        'layers':[{'layer':i,**metric(actual_layers[i][row],target_layers[i][0,0])} for i in sorted(actual_layers)]})
    for handle in handles:handle.remove()
    print(json.dumps({'diagnostic_only':True,'model_path':trace['path'],'checks':checks},indent=2))


if __name__=='__main__':main()

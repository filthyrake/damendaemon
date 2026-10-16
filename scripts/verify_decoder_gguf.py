"""Check a dense rotary decoder's GGUF comparator against dllm tensor values."""
import argparse
import json
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from dllm.qwen2 import ModelWeights
from dllm.quantization import dequantize as unpack


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('gguf')
    args=p.parse_args();torch.set_num_threads(4)
    model=ModelWeights(args.model,512,quantization='q8_0');c=model.config
    if c['model_type'] not in ('qwen2','qwen3'):p.error('unsupported comparison mapping')
    weights={k:(unpack(v,model.scales[k]) if k in model.scales else v.float()).numpy()
             for k,v in model.weights.items()}
    tensors={'token_embd.weight':weights['embed'],'output_norm.weight':weights['norm'],'output.weight':weights['head']}
    for layer in range(c['num_hidden_layers']):
        source=f'{layer}.';target=f'blk.{layer}.'
        for a,b in (('norm1','attn_norm.weight'),('norm2','ffn_norm.weight'),('out','attn_output.weight'),('down','ffn_down.weight')):
            tensors[target+b]=weights[source+a]
        for suffix in ('weight','bias'):
            name=source+'qkv.'+suffix
            if name not in weights:continue
            q=c['num_attention_heads']*c['head_dim'];kv=c['num_key_value_heads']*c['head_dim']
            for name,value in zip(('q','k','v'),np.split(weights[name],[q,q+kv])):
                tensors[target+'attn_'+name+'.'+suffix]=value
        gate,up=np.split(weights[source+'gate_up'],2)
        tensors[target+'ffn_gate.weight']=gate;tensors[target+'ffn_up.weight']=up
        for name in ('q','k'):
            if source+name+'_norm' in weights:tensors[target+'attn_'+name+'_norm.weight']=weights[source+name+'_norm']
    reader=GGUFReader(args.gguf);checks=[]
    for tensor in reader.tensors:
        expected=tensors.pop(tensor.name)
        actual=dequantize(tensor.data,tensor.tensor_type).reshape(expected.shape)
        if not np.array_equal(actual,expected):raise AssertionError((tensor.name,float(np.max(np.abs(actual-expected)))))
        checks.append({'tensor':tensor.name,'exact':True,'shape':list(expected.shape)})
    if tensors:raise ValueError('GGUF is missing expected tensors: '+', '.join(sorted(tensors)))
    print(json.dumps({'passed':True,'model':args.model,'gguf':args.gguf,'quantization':'q8_0','checks':checks},indent=2))


if __name__=='__main__':main()

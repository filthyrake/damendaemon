"""Independent Transformers FP32 reference, expanding only one layer at a time."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from tokenizers import Tokenizer
import transformers
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer,GptOssRotaryEmbedding,GptOssRMSNorm


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--prefix-lengths',type=int,nargs='+',default=[5,129])
    args=parser.parse_args();root=Path(args.model)
    if any(n<1 for n in args.prefix_lengths):raise ValueError('prefix lengths must be positive')
    device=args.device
    if device=='cuda':
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (
                'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283', 'GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a'):
            raise RuntimeError('CUDA reference restricted to the authorized A2/A30')
        torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    source=json.loads((root/'damendaemon-source.json').read_text());catalog=source['tensors']
    def get(key):
        with safe_open(str(root/catalog[key]['file']),framework='pt',device='cpu') as reader:
            return reader.get_tensor(key)
    def expand(blocks,scales):
        # Independent published E2M1 table + CPU FP64 exponent scaling.
        table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
        e,n,g,_=blocks.shape
        result=torch.empty(e,g*32,n)
        for i in range(e):
            codes=torch.stack((blocks[i]&15,blocks[i]>>4),-1).reshape(n,g,32).long()
            values=torch.ldexp(table[codes],(scales[i].int()-127)[...,None])
            if (scales[i]==255).any():raise ValueError('reserved scale in checkpoint')
            result[i].copy_(values.reshape(n,g*32).t())
        return result
    c=GptOssConfig.from_pretrained(root,local_files_only=True)
    c._attn_implementation='eager';c._experts_implementation='eager'
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    base=tokenizer.encode('The capital of France is Paris. Write a short explanation of how the ocean affects coastal weather. ',add_special_tokens=False).ids
    cases=[];embed=get('model.embed_tokens.weight').float().to(device)
    for split in args.prefix_lengths:
        length=split+8
        if length>c.max_position_embeddings:raise ValueError('reference exceeds checkpoint context')
        ids=(base*((length+len(base)-1)//len(base)))[:length]
        cases.append({'ids':ids,'split':split,'hidden':torch.nn.functional.embedding(torch.tensor(ids,device=device),embed)[None],
            'layers':[]})
    del embed
    rope=GptOssRotaryEmbedding(c).to(device)
    for item in cases:
        length=len(item['ids']);positions=torch.arange(length,device=device)[None]
        item['position_embeddings']=rope(item['hidden'],positions)
    start=time.perf_counter()
    for i in range(c.num_hidden_layers):
        with torch.device('meta'):layer=GptOssDecoderLayer(c,i)
        state={};prefix=f'model.layers.{i}.'
        for key in catalog:
            if key.startswith(prefix) and not key.endswith(('_blocks','_scales')):
                state[key[len(prefix):]]=get(key).float()
        for projection in ('gate_up_proj','down_proj'):
            key=prefix+'mlp.experts.'+projection
            state['mlp.experts.'+projection]=expand(get(key+'_blocks'),get(key+'_scales'))
        layer.load_state_dict(state,strict=True,assign=True);layer.to(device);layer.eval();del state
        for item in cases:
            length=len(item['ids']);pos=torch.arange(length,device=device)
            valid=pos[None,:]<=pos[:,None]
            if c.layer_types[i]=='sliding_attention':valid &= pos[None,:]>pos[:,None]-c.sliding_window
            mask=torch.zeros((length,length),device=device).masked_fill(~valid,float('-inf'))[None,None]
            hidden=layer(item['hidden'],attention_mask=mask,position_embeddings=item['position_embeddings'])
            item['hidden']=hidden;item['layers'].append(hidden[0].cpu().clone())
        del layer
        print(json.dumps({'layer':i,'seconds':time.perf_counter()-start}),flush=True)
    norm=GptOssRMSNorm(c.hidden_size,eps=c.rms_norm_eps)
    norm.load_state_dict({'weight':get('model.norm.weight').float()})
    norm.to(device)
    head=get('lm_head.weight').float().to(device)
    for item in cases:
        # Logits at split-1 predict the first continuation token. Subsequent
        # entries qualify teacher-forced decode against a full causal reference.
        item['logits']=torch.nn.functional.linear(norm(item['hidden'][:,item['split']-1:]),head)[0].cpu()
        del item['hidden'],item['position_embeddings']
    report={'repository':source['repository'],'revision':source['revision'],'transformers':transformers.__version__,
        'torch':str(torch.__version__),'dtype':'float32','device':device,'float32_matmul_precision':torch.get_float32_matmul_precision(),'attention':'eager','cases':cases,
        'reference':f'Unmodified Transformers decoder layers; independent table/FP64 unpack, layer-streamed {device} weights'}
    torch.save(report,args.output)
    print(json.dumps({'passed':True,'output':args.output,'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':main()

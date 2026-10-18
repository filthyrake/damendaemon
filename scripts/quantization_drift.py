"""Small fixed-corpus weight-quantization drift study; not a quality benchmark."""
import argparse
import gc
import json
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer
from reference_model import load_reference

TEXTS=[
    'The Earth travels around the Sun once every year. The Moon travels around the Earth.',
    'A binary search repeatedly divides a sorted list in half until it finds the requested value.',
    'Water freezes at zero degrees Celsius under ordinary atmospheric pressure.',
    'A gardener planted tomato seeds in spring and watered them every morning.',
    'The sum of seven and five is twelve. Multiplying three by four also gives twelve.',
    'In Python, a dictionary maps keys to values, while a list stores an ordered sequence of elements.',
    'The train left the station at noon and arrived in the next city two hours later.',
    'Bonjour, comment allez-vous? Je voudrais un café et un verre d’eau, s’il vous plaît.',
    '東京は日本の首都です。今日は空が青く、穏やかな風が吹いています。',
    'A reliable program validates its inputs, handles errors, and keeps unrelated requests isolated.'
]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--format',choices=('q8_0','q4_0'),default='q4_0')
    p.add_argument('--weight-dtype',choices=('float16','float32'),default='float16')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--arithmetic',choices=('float32','float16'),default='float32',
                   help='cpu arithmetic; --device cuda always uses float16')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    inputs=[torch.tensor([tokenizer.encode(text,add_special_tokens=True)]) for text in TEXTS]
    baseline=[];reports=[]
    kind=json.loads((Path(args.model)/'config.json').read_text())['model_type']
    load_kwargs: dict = dict(reference_quantization='none',local_files_only=True,
        trust_remote_code=False,attn_implementation='eager',
        **({'chunk_size':16} if kind=='mamba2' else {}))
    if args.device=='cuda':
        # Bounded-device mode: FP16 weights with device_map auto; arithmetic
        # stays FP16 (declared in the report). The CPU default keeps FP32.
        load_kwargs.update(device_map='auto',torch_dtype=torch.float16)
        arithmetic='float16'
    else:
        load_kwargs.update(torch_dtype=getattr(torch,args.weight_dtype))
        arithmetic=args.arithmetic
        if arithmetic=='float16' and args.weight_dtype!='float16':
            raise ValueError('float16 cpu arithmetic requires --weight-dtype float16')
    for quantization in ('none',args.format):
        load_kwargs['reference_quantization']=quantization
        model=load_reference(args.model,**load_kwargs)
        if arithmetic=='float32':model=model.float()
        model.eval()
        nll=0.;tokens=0;matches=0;relative=[]
        for index,ids in enumerate(inputs):
            logits=model(ids,use_cache=False).logits[:,:-1].float()
            nll+=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten(),reduction='sum').item()
            tokens+=ids.numel()-1
            if quantization=='none':baseline.append(logits.cpu().clone())
            else:
                expected=baseline[index]
                matches+=(logits.cpu().argmax(-1)==expected.argmax(-1)).sum().item()
                relative.append(((logits.cpu()-expected).square().mean()/expected.square().mean()).sqrt().item())
        reports.append({'weight_quantization':quantization,'tokens':tokens,'mean_nll':nll/tokens,
            'perplexity':math.exp(nll/tokens),'teacher_forced_top1_agreement':matches/tokens if quantization!='none' else None,
            'per_text_relative_logit_rms':relative})
        del model;gc.collect()
    print(json.dumps({'model':args.model,'device':args.device,
        'arithmetic':arithmetic,'weight_conversion_dtype':args.weight_dtype,
        'threads':args.threads,'texts':TEXTS,
        'scope':'small authored corpus; measures quantization drift, not broad task quality or GPU performance',
        'results':reports},indent=2))


if __name__=='__main__':main()

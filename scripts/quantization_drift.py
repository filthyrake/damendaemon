"""Small fixed-corpus weight-quantization drift study; not a quality benchmark."""
import argparse
import gc
import json
import math
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
    args=p.parse_args()
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    inputs=[torch.tensor([tokenizer.encode(text,add_special_tokens=True)]) for text in TEXTS]
    baseline=[];reports=[]
    for quantization in ('none',args.format):
        model=load_reference(args.model,reference_quantization=quantization,local_files_only=True,
            trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='eager').float().eval()
        nll=0.;tokens=0;matches=0;relative=[]
        for index,ids in enumerate(inputs):
            logits=model(ids,use_cache=False).logits[:,:-1].float()
            nll+=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten(),reduction='sum').item()
            tokens+=ids.numel()-1
            if quantization=='none':baseline.append(logits.clone())
            else:
                expected=baseline[index]
                matches+=(logits.argmax(-1)==expected.argmax(-1)).sum().item()
                relative.append(((logits-expected).square().mean()/expected.square().mean()).sqrt().item())
        reports.append({'weight_quantization':quantization,'tokens':tokens,'mean_nll':nll/tokens,
            'perplexity':math.exp(nll/tokens),'teacher_forced_top1_agreement':matches/tokens if quantization!='none' else None,
            'per_text_relative_logit_rms':relative})
        del model;gc.collect()
    print(json.dumps({'model':args.model,'device':'cpu','arithmetic':'float32','weight_conversion_dtype':'float16','texts':TEXTS,
        'scope':'small authored corpus; measures quantization drift, not broad task quality or GPU performance',
        'results':reports},indent=2))


if __name__=='__main__':main()

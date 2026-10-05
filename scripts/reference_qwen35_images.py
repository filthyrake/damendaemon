"""Independent image/text prefill and teacher-forced decode targets."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, Qwen3_5ForConditionalGeneration


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('vision_reference')
    parser.add_argument('--output',required=True)
    parser.add_argument('--long-context',action='store_true',help='near-1024-token image prompt for recurrent-state validation')
    parser.add_argument('--text-quantization',choices=('none','q8_0'),default='none')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    fixture=torch.load(args.vision_reference,weights_only=True)
    path=fixture['model']
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    reference=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.float16,
        attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
    # Separate vision FP32 arithmetic from the established FP16 text policy.
    reference.model.visual.float()
    if args.text_quantization=='q8_0':
        from gguf.quants import quantize,dequantize
        from gguf.constants import GGMLQuantizationType
        # Only text projections; preserve the independent vision precision policy.
        projections=[module for module in reference.model.language_model.modules()
                     if isinstance(module,torch.nn.Linear)]+[reference.lm_head]
        for module in projections:
            values=module.weight.detach().float().numpy()
            restored=dequantize(quantize(values,GGMLQuantizationType.Q8_0),GGMLQuantizationType.Q8_0)
            module.weight=torch.nn.Parameter(torch.from_numpy(restored).half(),requires_grad=False)
    reference.cuda()
    artifact=dict(model=path,cases=[],vision_dtype='float32',text_dtype='float16',steps=8,
                  text_quantization=args.text_quantization,
                  vision_weight_policy='loaded FP16 then promoted to FP32',
                  torch=str(torch.__version__),transformers=transformers.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cases=[case for case in fixture['cases'] if case['name'] in ('square','resized','two-images')]
    processor=AutoImageProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    square=next(case for case in cases if case['name']=='square')
    pair=next(case for case in cases if case['name']=='two-images')
    for name,images in (('square-inverted',[255-square['images'][0]]),
                        ('two-images-reversed',list(reversed(pair['images'])))):
        prepared=processor(images=[Image.fromarray(image.numpy()) for image in images],return_tensors='pt')
        cases.append(dict(name=name,images=images,pixel_values=prepared.pixel_values,
                          grids=prepared.image_grid_thw.tolist()))
    cases.append(dict(name='text-only-reused-slot',images=[],grids=[]))
    if args.long_context:
        cases=[{**square,'name':'long-square','text':'color '*900+'Describe the colors and shapes in these images.'}]
    for case in cases:
        content=''.join('<|vision_start|><|image_pad|><|vision_end|>' for _ in case['grids'])
        text=case.get('text','Describe the colors and shapes in these images.')
        content+=text
        prompt=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if isinstance(prompt,dict):prompt=prompt['input_ids']
        tokens=[]
        image=0
        for token in prompt:
            if token==248056:
                _,h,w=case['grids'][image]
                tokens.extend([token]*(h*w//4));image+=1
            else:tokens.append(token)
        ids=torch.tensor([tokens],device='cuda')
        image_args=dict(pixel_values=case['pixel_values'].cuda(),
            image_grid_thw=torch.tensor(case['grids'],device='cuda')) if case['grids'] else {}
        reference.model.rope_deltas=None
        result=reference(input_ids=ids,**image_args,
            mm_token_type_ids=(ids==248056).int(),use_cache=True,logits_to_keep=1)
        target=result.logits[:,-1].float().cpu()
        entry=dict(name=case['name'],text=text,tokens=tokens,images=case['images'],grids=case['grids'],
                   targets=[target],decode_tokens=[])
        for _ in range(8):
            token=target.argmax().item()
            entry['decode_tokens'].append(token)
            result=reference(input_ids=torch.tensor([[token]],device='cuda'),
                past_key_values=result.past_key_values,use_cache=True,logits_to_keep=1)
            target=result.logits[:,-1].float().cpu()
            entry['targets'].append(target)
        artifact['cases'].append(entry)
        print(json.dumps(dict(name=case['name'],tokens=len(tokens),decode=entry['decode_tokens'])),flush=True)
    torch.save(artifact,args.output)


if __name__=='__main__':main()

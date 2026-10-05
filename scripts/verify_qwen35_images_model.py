"""Experimental end-to-end image/text logits against independent targets."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from dllm.qwen2 import ModelWeights
from dllm.vision_qwen35 import VisionWeights, VisionEncoder
from dllm.vision_inputs import prepare_images
from experiments.vision_qwen35 import ImageHybridExecutor


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    reference=torch.load(args.reference,weights_only=True)
    model=ModelWeights(reference['model'],1024)
    executor=ImageHybridExecutor(model,max_batch=3)
    vision=VisionEncoder(VisionWeights(reference['model']))
    report=dict(passed=False,scope='experimental multimodal computation; not serving qualification',
                device=torch.cuda.get_device_name(),torch=str(torch.__version__),checks=[],
                reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                    ('dllm/vision_qwen35.py','dllm/vision_inputs.py','dllm/vision_positions.py',
                     'dllm/hybrid.py','scripts/experiments/vision_qwen35.py')})
    def check(actual,target,label):
        actual=actual.float().cpu().reshape(-1)
        target=target.reshape(-1)
        difference=actual-target
        rms=(difference.square().mean()/target.square().mean().clamp_min(1e-12)).sqrt().item()
        maximum=difference.abs().max().item()
        match=actual.argmax().item()==target.argmax().item()
        record=dict(label=label,relative_rms=rms,max_absolute=maximum,greedy_match=match,
            passed=bool(torch.isfinite(actual).all()) and rms<.01 and maximum<.5 and match)
        report['checks'].append(record)
        print(json.dumps(record),flush=True)
    try:
        prefills={}
        for start in range(0,len(reference['cases']),3):
            cases=reference['cases'][start:start+3]
            for slot,case in enumerate(cases):
                if case['grids']:
                    pixels,grids=prepare_images(case['images'])
                    features=vision.encode(pixels,grids)
                    actual=executor.prefill_images(case['tokens'],features,grids,slot)
                else:
                    actual=executor.prefill(case['tokens'],slot)
                check(actual,case['targets'][0],case['name']+':prefill')
                prefills[case['name']]=actual.float().cpu().clone()
            # Reordered slots exercise independent image offsets during batched decode.
            slots=list(reversed(range(len(cases))))
            for step in range(reference['steps']):
                tokens=[cases[slot]['decode_tokens'][step] for slot in slots]
                actual,_=executor.decode_batch(tokens,slots)
                for row,slot in enumerate(slots):
                    case=cases[slot]
                    check(actual[row],case['targets'][step+1],case['name']+f':decode:{step}')
        report['passed']=all(check['passed'] for check in report['checks'])
        if 'square-inverted' in prefills:
            cases={case['name']:case for case in reference['cases']}
            original,changed=cases['square'],cases['square-inverted']
            report['changed_image_control']=dict(same_tokens=original['tokens']==changed['tokens'],
                native_logit_difference=(prefills['square']-prefills['square-inverted']).abs().max().item(),
                reference_logit_difference=(original['targets'][0]-changed['targets'][0]).abs().max().item())
            control=report['changed_image_control']
            report['passed'] &= (control['same_tokens'] and control['native_logit_difference']>0
                                 and control['reference_logit_difference']>0)
        if not report['passed']:raise AssertionError('independent image/text model gate failed')
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

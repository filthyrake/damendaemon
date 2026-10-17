"""Warm experimental image-to-text timing, not an HTTP benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from damendaemon.qwen2 import ModelWeights
from damendaemon.vision_qwen35 import VisionWeights, VisionEncoder
from damendaemon.vision_inputs import prepare_images
from experiments.vision_qwen35 import ImageHybridExecutor


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('--output',required=True)
    parser.add_argument('--graph',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    reference=torch.load(args.reference,weights_only=True)
    executor=ImageHybridExecutor(ModelWeights(reference['model'],1024,
        quantization=reference.get('text_quantization','none')),max_batch=1,use_graph=args.graph)
    vision=VisionEncoder(VisionWeights(reference['model']))
    report=dict(scope='experimental warm image-to-text, no HTTP, no model load, fixed 32 tokens',
        graph=args.graph,
        text_quantization=reference.get('text_quantization','none'),
        text_host_weight_bytes=executor.model.bytes,vision_host_weight_bytes=vision.model.bytes,
        device=torch.cuda.get_device_name(),torch=str(torch.__version__),model=reference['model'],cases=[],
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('damendaemon/vision_qwen35.py','damendaemon/vision_inputs.py','damendaemon/vision_positions.py',
             'damendaemon/hybrid.py','damendaemon/vision_hybrid.py','scripts/experiments/vision_qwen35.py')},
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest())
    for case in reference['cases'][:3]:
        trials=[]
        outputs=[]
        for trial in range(4):
            torch.cuda.synchronize()
            start=time.perf_counter()
            pixels,grids=prepare_images(case['images'])
            processed=time.perf_counter()
            features=vision.encode(pixels,grids)
            torch.cuda.synchronize()
            encoded=time.perf_counter()
            logits=executor.prefill_images(case['tokens'],features,grids)
            token=logits.argmax().item()
            output=[token]
            first=time.perf_counter()
            for _ in range(31):
                logits,_=executor.decode_batch([token],[0])
                token=logits.argmax().item()
                output.append(token)
            end=time.perf_counter()
            if output[:8]!=case['decode_tokens']:
                raise AssertionError('benchmark generation differs from independent first-eight-token reference')
            if trial:
                trials.append(dict(preprocessing_seconds=processed-start,vision_seconds=encoded-processed,
                    text_prefill_seconds=first-encoded,time_to_first_token_seconds=first-start,
                    decode_seconds=end-first,total_seconds=end-start))
                outputs.append(output)
        if any(output!=outputs[0] for output in outputs):raise AssertionError('unstable benchmark output')
        result=dict(name=case['name'],prompt_tokens=len(case['tokens']),output_tokens=32,
                    trials=trials,generated_ids=outputs[0],median={key:statistics.median(t[key] for t in trials)
                        for key in trials[0]})
        report['cases'].append(result)
        print(json.dumps(result),flush=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

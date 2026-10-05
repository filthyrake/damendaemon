"""A2 image encoder graph candidate: changed-pixel proof and warm ABBA timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from dllm.vision_inputs import prepare_images
from dllm.vision_qwen35 import VisionWeights,VisionEncoder
from experiments.vision_encoder_graph import GraphVisionEncoder


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    reference=torch.load(args.reference,weights_only=True)
    encoder=VisionEncoder(VisionWeights(reference['model']))
    graph=GraphVisionEncoder(encoder)
    report=dict(passed=False,scope='experimental encoder only; CPU packed pixels, includes transfer; excludes preprocessing/text/HTTP',
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('dllm/vision_qwen35.py','scripts/experiments/vision_encoder_graph.py')},cases=[])
    try:
        for case in reference['cases']:
            if not case['images']:continue
            pixels,grids=prepare_images(case['images'])
            target=encoder.encode(pixels,grids).clone()
            torch.cuda.synchronize();start=time.perf_counter()
            actual=graph.encode(pixels,grids);torch.cuda.synchronize()
            capture=time.perf_counter()-start
            error=actual-target
            rms=(error.square().mean()/target.square().mean().clamp_min(1e-12)).sqrt().item()
            maximum=error.abs().max().item()
            assert torch.isfinite(actual).all() and rms<1e-4 and maximum<.02,case['name']
            changed_target=encoder.encode(-pixels,grids).clone()
            changed=graph.encode(-pixels,grids)
            changed_error=changed-changed_target
            changed_rms=(changed_error.square().mean()/changed_target.square().mean().clamp_min(1e-12)).sqrt().item()
            changed_max=changed_error.abs().max().item()
            assert torch.isfinite(changed).all() and changed_rms<1e-4 and changed_max<.02
            assert (changed-target).abs().max().item()>.01, 'changed pixels did not change features'
            phases=[]
            for mode in ('eager','graph','graph','eager'):
                fn=encoder.encode if mode=='eager' else graph.encode
                for _ in range(2):fn(pixels,grids)
                torch.cuda.synchronize();times=[]
                for _ in range(10):
                    start=time.perf_counter();fn(pixels,grids);torch.cuda.synchronize()
                    times.append(time.perf_counter()-start)
                phases.append(dict(mode=mode,seconds=times,median_seconds=statistics.median(times)))
            report['cases'].append(dict(name=case['name'],grids=grids,relative_rms=rms,max_absolute=maximum,
                changed_pixel_relative_rms=changed_rms,changed_pixel_max_absolute=changed_max,
                first_graph_seconds=capture,phases=phases,cuda_allocated_bytes=torch.cuda.memory_allocated()))
        report['passed']=True
    except BaseException as error:
        report['error']=repr(error);raise
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

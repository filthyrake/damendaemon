"""Independent intermediate-tensor gate for damendaemon's standalone image encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from damendaemon.vision_qwen35 import VisionWeights, VisionEncoder


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',help='trusted locally generated vision reference')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('cuda','mps'),default='cuda')
    parser.add_argument('--model',help='local copy of the pinned reference checkpoint')
    parser.add_argument('--kernels',choices=('auto','torch'),default='auto')
    parser.add_argument('--attention',choices=('sdpa','eager'),default='sdpa')
    parser.add_argument('--source-weights',action='store_true',help='preserve source coefficients in FP32 storage')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    reference=torch.load(args.reference,map_location='cpu',weights_only=True)
    model=VisionWeights(args.model or reference['model'],projection_dtype=torch.float32 if args.source_weights else torch.float16)
    encoder=VisionEncoder(model,device=args.device,kernels=args.kernels,attention=args.attention)
    synchronize=torch.mps.synchronize if args.device=='mps' else torch.cuda.synchronize
    allocated=torch.mps.current_allocated_memory if args.device=='mps' else torch.cuda.memory_allocated
    report=dict(passed=False,scope='standalone vision encoder, not multimodal serving qualification',
                device='Apple MPS' if args.device=='mps' else torch.cuda.get_device_name(),torch=str(torch.__version__),backend=encoder.backend,
                attention=encoder.attention,
                projection_storage=str(model.projection_dtype),
                model=str(model.path),reference_model=reference['model'],host_weight_bytes=model.bytes,device_weight_bytes=encoder.bytes,
                reference_device=reference['reference_device'],reference_dtype=reference['reference_dtype'],
                reference_transformers=reference['transformers'],reference_torch=reference['torch'],
                reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path('damendaemon/vision_qwen35.py').read_bytes()).hexdigest(),
                gates=dict(relative_rms=1e-4,max_absolute=.02),cases=[])
    try:
        for case in reference['cases']:
            trace={}
            pixels=case['pixel_values'].to(args.device)
            encoder.encode(pixels,case['grids'],trace=trace)
            checks=[]
            for name,target in case['targets'].items():
                actual=trace[name]
                if actual.shape!=target.shape:raise AssertionError('encoder stage shape differs')
                error=actual-target
                rms=(error.square().mean()/target.square().mean().clamp_min(1e-12)).sqrt().item()
                maximum=error.abs().max().item()
                checks.append(dict(stage=name,relative_rms=rms,max_absolute=maximum,
                    passed=bool(torch.isfinite(actual).all()) and rms<1e-4 and maximum<.02))
            result=dict(name=case['name'],grids=case['grids'],checks=checks)
            report['cases'].append(result)
            if not all(check['passed'] for check in checks):
                print(json.dumps(result),flush=True)
                continue
            for _ in range(3):encoder.encode(pixels,case['grids'])
            synchronize()
            baseline=allocated()
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(5):
                start=time.perf_counter()
                encoder.encode(pixels,case['grids'])
                synchronize()
                times.append(time.perf_counter()-start)
            result.update(median_encoder_seconds=statistics.median(times),encoder_seconds=times,
                          peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-baseline if args.device=='cuda' else None,
                          timing_scope='warm encoder only; staged GPU pixels; input processing/copy and text decoding excluded')
            print(json.dumps(result),flush=True)
        report['passed']=all(check['passed'] for case in report['cases'] for check in case['checks'])
        if not report['passed']:raise AssertionError('independent vision gate failed')
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

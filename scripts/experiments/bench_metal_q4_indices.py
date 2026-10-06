"""Counterbalanced full decode comparison of Q4 index arithmetic on MPS."""
import argparse
import json
import statistics
import time
import torch
from dllm import quantized_metal_q4
from dllm.models import load
from dllm.runtime import make_executor


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model');parser.add_argument('prompt')
    parser.add_argument('--batch',type=int,choices=(1,2,4),default=1)
    args=parser.parse_args();torch.set_num_threads(4)
    model=load(args.model,512)
    executor=make_executor(model,max_batch=args.batch)
    assert executor.device=='mps' and model.quantization=='q4_0'
    prompt=json.load(open(args.prompt));slots=list(range(args.batch))
    original=quantized_metal_q4.library;shader=original()
    class Wide:
        def __getattr__(self,name):return getattr(shader,name.removesuffix('_index32'))
    wide=Wide();results=[];expected=None
    def run(label):
        quantized_metal_q4.library=(lambda:wide) if label=='wide' else original
        first=executor.prefill_batch([prompt]*args.batch,slots).argmax(-1).cpu().tolist()
        tokens=first;sequence=[first];torch.mps.synchronize();start=time.perf_counter()
        for _ in range(31):
            _,ids=executor.decode_batch(tokens,slots)
            tokens=ids.cpu().tolist();sequence.append(tokens)
        elapsed=time.perf_counter()-start
        return dict(mode=label,decode_seconds=elapsed,aggregate_decode_tokens_per_second=31*args.batch/elapsed,tokens=sequence)
    try:
        for label in ('wide','index32'):run(label)
        for label in ('wide','index32','index32','wide')*3:
            result=run(label)
            if expected is None:expected=result['tokens']
            assert result['tokens']==expected
            results.append(result)
    finally:quantized_metal_q4.library=original
    print(json.dumps(dict(passed=True,batch=args.batch,context=512,prompt=prompt,output_tokens=32,
        device='Apple M4',torch=str(torch.__version__),quantization='q4_0',
        scope='Same weights and floating-point arithmetic; warm decode includes token readback, excludes prefill and HTTP',
        median_decode_tokens_per_second={label:statistics.median(r['aggregate_decode_tokens_per_second'] for r in results if r['mode']==label) for label in ('wide','index32')},
        trials=results),indent=2))


if __name__=='__main__':main()

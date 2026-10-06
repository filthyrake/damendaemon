"""Counterbalanced full-model short-prefill experiment; no runtime policy changes."""
import argparse
import json
import statistics
import time
import torch
from dllm import quantized_metal,quantized_metal_q4
from dllm.models import load
from dllm.runtime import make_executor


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt')
    p.add_argument('--quantization',choices=('q8_0','q4_0'),default='q8_0')
    p.add_argument('--batch',type=int,choices=(1,4),default=1);args=p.parse_args()
    torch.set_num_threads(4);model=load(args.model,512)
    executor=make_executor(model,max_batch=args.batch)
    assert executor.device=='mps' and model.quantization==args.quantization
    module=quantized_metal if args.quantization=='q8_0' else quantized_metal_q4
    original=module.linear;shader=module.library()
    original_policy=module.use_packed_projection
    module.use_packed_projection=lambda rows,outputs,width,fp32:rows<=4
    def candidate(x,w,scales,bias=None):
        width=x.shape[-1];rows=x.numel()//width
        if x.dtype!=torch.float32 or rows<=4 or rows>16 or (rows>8 and w.shape[0]*width<4000000):
            return original(x,w,scales,bias)
        shape=x.shape[:-1];x=x.reshape(rows,width).contiguous()
        if bias is not None:bias=bias.float().contiguous()
        out=torch.empty((rows,w.shape[0]),device=x.device,dtype=x.dtype)
        kernel=shader.linear_index32 if args.quantization=='q8_0' else shader.linear_float_index32
        kernel(x,w,scales,x if bias is None else bias,out,rows,w.shape[0],width,int(bias is not None),
            threads=(((w.shape[0]+7)//8)*256,(rows+3)//4),group_size=(256,1))
        return out.reshape(*shape,w.shape[0])
    ids=json.load(open(args.prompt));results=[]
    try:
        for length in (8,16,32):
            prompt=(ids*4)[:length];slots=list(range(args.batch))
            module.linear=original
            expected=executor.prefill_batch([prompt]*args.batch,slots).cpu()
            def run(mode):
                module.linear=original if mode=='expanded' else candidate
                torch.mps.synchronize();start=time.perf_counter()
                actual=executor.prefill_batch([prompt]*args.batch,slots)
                tokens=actual.argmax(-1).cpu().tolist();elapsed=time.perf_counter()-start
                actual=actual.cpu();error=actual-expected
                relative=(error.square().mean()/expected.square().mean()).sqrt().item()
                assert relative<.001 and torch.isfinite(actual).all()
                assert tokens==expected.argmax(-1).tolist()
                return dict(mode=mode,seconds=elapsed,relative_rms=relative,max_absolute=error.abs().max().item(),tokens=tokens)
            for mode in ('expanded','packed'):run(mode)
            trials=[run(mode) for mode in ('expanded','packed','packed','expanded')*3]
            results.append(dict(prompt=prompt,batch=args.batch,trials=trials,
                median_ms={mode:1000*statistics.median(r['seconds'] for r in trials if r['mode']==mode) for mode in ('expanded','packed')}))
    finally:
        module.linear=original
        module.use_packed_projection=original_policy
    print(json.dumps(dict(passed=True,device='Apple M4',quantization=args.quantization,context=512,
        scope='Same coefficients; FP32 packed versus MPS matmul accumulation differs. Prefill plus first token readback only; not HTTP.',results=results),indent=2))


if __name__=='__main__':main()

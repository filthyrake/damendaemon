"""Measure Q4 SIMD-group packing with identical coefficients and arithmetic."""
import json
import statistics
import time
import torch
from dllm.quantized_metal_q4 import library


@torch.inference_mode()
def main():
    torch.manual_seed(419);torch.set_num_threads(4)
    shader=library();results=[]
    for outputs,width in ((1152,896),(896,896),(9728,896),(896,4864),(151936,896)):
        w=torch.randint(0,256,(outputs,width//2),dtype=torch.uint8,device='mps')
        scales=torch.full((outputs,width//32),.01,dtype=torch.float16,device='mps')
        for rows in (1,4):
            x=torch.randn(rows,width,device='mps')
            out=torch.empty((rows,outputs),device='mps')
            def run(group):
                channels=group//32
                shader.linear_float(x,w,scales,x,out,rows,outputs,width,0,
                    threads=(((outputs+channels-1)//channels)*group,1),group_size=(group,1))
            run(256);expected=out.clone();torch.mps.synchronize()
            timings={group:[] for group in (64,128,256,512)}
            for group in (256,64,128,512,512,128,64,256)*3:
                run(group);torch.testing.assert_close(out,expected,atol=0,rtol=0)
                for _ in range(3):run(group)
                torch.mps.synchronize();start=time.perf_counter()
                for _ in range(20):run(group)
                torch.mps.synchronize();timings[group].append((time.perf_counter()-start)*1000/20)
            results.append(dict(rows=rows,outputs=outputs,width=width,
                median_ms={k:statistics.median(v) for k,v in timings.items()},trials_ms=timings))
    print(json.dumps(dict(passed=True,device='Apple M4',torch=str(torch.__version__),
        scope='Same Q4 coefficients and arithmetic; dispatch plus GPU completion, not full serving',results=results),indent=2))


if __name__=='__main__':main()

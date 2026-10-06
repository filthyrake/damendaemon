"""Measure Q8 32-bit index arithmetic with identical coefficients and arithmetic."""
import json
import statistics
import time
import torch
from dllm.quantized_metal import library, _WIDE_SOURCE


@torch.inference_mode()
def main():
    torch.manual_seed(419);torch.set_num_threads(4)
    shader=library();results=[]
    source=_WIDE_SOURCE
    for name in ('rows','outputs','width','biased'):
        source=source.replace('constant long& '+name, 'constant long& '+name+'_arg')
    source=source.replace('long channel=id.x/32,batch=id.y*4;',
        'uint rows=uint(rows_arg), outputs=uint(outputs_arg), width=uint(width_arg), biased=uint(biased_arg); uint channel=id.x/32, batch=id.y*4;')
    source=source.replace('for(long d=lane;', 'for(uint d=lane;').replace('long low=', 'uint low=')
    candidate=torch.mps.compile_shader(source)
    for outputs,width in ((1152,896),(896,896),(9728,896),(896,4864),(151936,896)):
        w=torch.randint(-127,128,(outputs,width),dtype=torch.int8,device='mps')
        scales=torch.full((outputs,width//32),.01,dtype=torch.float16,device='mps')
        for rows in (1,4):
            x=torch.randn(rows,width,device='mps')
            out=torch.empty((rows,outputs),device='mps')
            def run(label):
                group=256
                channels=group//32
                selected=shader if label=='baseline' else candidate
                selected.linear(x,w,scales,x,out,rows,outputs,width,0,
                    threads=(((outputs+channels-1)//channels)*group,1),group_size=(group,1))
            run('baseline');expected=out.clone();torch.mps.synchronize()
            timings={group:[] for group in ('baseline','uint32')}
            for group in ('baseline','uint32','uint32','baseline')*3:
                run(group);torch.testing.assert_close(out,expected,atol=0,rtol=0)
                for _ in range(3):run(group)
                torch.mps.synchronize();start=time.perf_counter()
                for _ in range(20):run(group)
                torch.mps.synchronize();timings[group].append((time.perf_counter()-start)*1000/20)
            results.append(dict(rows=rows,outputs=outputs,width=width,
                median_ms={k:statistics.median(v) for k,v in timings.items()},trials_ms=timings))
    print(json.dumps(dict(passed=True,device='Apple M4',torch=str(torch.__version__),
        scope='Same Q8 coefficients and floating-point arithmetic; 32-bit index experiment; dispatch plus GPU completion, not full serving',results=results),indent=2))


if __name__=='__main__':main()

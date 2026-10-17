"""Counterbalanced native Q8/FP32 projection timings with identical weights."""
import json
import statistics
import time
import torch
from damendaemon import metal_kernels,quantized_metal


@torch.inference_mode()
def main():
    torch.manual_seed(419);torch.set_num_threads(4);results=[]
    for outputs,width in ((3336,768),(768,1536),(50288,768)):
        q=torch.randint(-127,128,(outputs,width),dtype=torch.int8,device='mps')
        scales=torch.full((outputs,width//32),.01,dtype=torch.float16,device='mps')
        full=(q.float().reshape(outputs,width//32,32)*scales.float()[:,:,None]).reshape(outputs,width)
        for rows in (1,4):
            x=torch.randn(rows,width,device='mps')
            expected=metal_kernels.linear(x,full);actual=quantized_metal.linear(x,q,scales)
            torch.testing.assert_close(actual,expected,atol=.001,rtol=.001)
            timings={'fp32':[],'q8':[]}
            for label in ('fp32','q8','q8','fp32')*3:
                fn=(lambda:metal_kernels.linear(x,full)) if label=='fp32' else (lambda:quantized_metal.linear(x,q,scales))
                for _ in range(3):fn()
                torch.mps.synchronize();start=time.perf_counter()
                for _ in range(20):fn()
                torch.mps.synchronize();timings[label].append((time.perf_counter()-start)*1000/20)
            results.append({'rows':rows,'outputs':outputs,'width':width,
                'median_ms':{k:statistics.median(v) for k,v in timings.items()},'trials_ms':timings})
    print(json.dumps({'passed':True,'device':'Apple M4','results':results,
        'scope':'same dequantized values; includes native dispatch and GPU completion, not full serving'},indent=2))


if __name__=='__main__':main()

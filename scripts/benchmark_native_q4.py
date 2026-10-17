"""Profile experimental CUDA C++ Q4 GEMV against independently dequantized math."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import ctypes
import json
import torch
import triton.testing
from damendaemon.runtime import require_gpu
from damendaemon.quantization import quantize_q4,dequantize_q4
from damendaemon.quantized_kernels_q4 import linear


def main():
    p=argparse.ArgumentParser();p.add_argument('library');p.add_argument('--output',default='q4-native-tuning.json');args=p.parse_args()
    require_gpu();torch.set_num_threads(4);torch.manual_seed(97)
    lib=ctypes.CDLL(args.library)
    fn=lib.damendaemon_q4_gemv
    fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
    fn.restype=ctypes.c_int
    records=[]
    for n,k in ((113,96),(6144,4096),(28672,4096),(4096,14336),(32000,4096),(151936,896)):
        q,s=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
        w=dequantize_q4(q,s,torch.float16)
        for rows in (1,2,4):
            x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
            bias=torch.randn(n,device='cuda',dtype=torch.float16)
            y=torch.empty((rows,n),device='cuda',dtype=torch.float16)
            expected=torch.nn.functional.linear(x,w,bias)
            def launch(with_bias=True):
                error=fn(x.data_ptr(),q.data_ptr(),s.data_ptr(),bias.data_ptr() if with_bias else None,y.data_ptr(),rows,n,k,torch.cuda.current_stream().cuda_stream)
                if error:raise RuntimeError(f'CUDA native Q4 launch failed: {error}')
            launch(False)
            no_bias=torch.nn.functional.linear(x,w)
            no_bias_relative=((y.float()-no_bias.float()).square().mean()/no_bias.float().square().mean()).sqrt().item()
            assert no_bias_relative<.001,(n,k,rows,no_bias_relative)
            launch()
            relative=((y.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
            assert relative<.001,(n,k,rows,relative)
            record={'outputs':n,'width':k,'rows':rows,'relative_rms':relative,'no_bias_relative_rms':no_bias_relative,
                    'triton_ms':triton.testing.do_bench_cudagraph(lambda:linear(x,q,s,bias),rep=100),
                    'cuda_cpp_ms':triton.testing.do_bench_cudagraph(launch,rep=100)}
            records.append(record);print(json.dumps(record),flush=True)
    open(args.output,'w').write(json.dumps({'device':torch.cuda.get_device_name(),'checks':records},indent=2))


if __name__=='__main__':main()

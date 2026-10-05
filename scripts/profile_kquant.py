"""Time actual packed projections with CUDA events; never requantize coefficients."""
import argparse
import json
import os
import torch
import triton as tr
from dllm.gguf_model import GgufWeights
from dllm.kquant_kernels import _gemv, _gemm, expand, linear, split_gemv


def timing(fn):
    for _ in range(3): fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output=fn()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/10,output


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--decode-tuning',action='store_true')
    p.add_argument('--native-only',action='store_true')
    p.add_argument('--residual',action='store_true',help='include isolated residual-int8 candidate')
    args=p.parse_args()
    if args.residual:args.native_only=True
    torch.set_num_threads(4);torch.manual_seed(613)
    torch.backends.cuda.matmul.allow_tf32=False
    model=GgufWeights(args.model,512,'none')
    with torch.inference_mode():
        for name in ('0.qkv.weight','0.out','0.gate_up','0.down','head'):
            weight=model.weights[name].cuda();layout=model.packed_formats[name]
            width,spec=layout;outputs=spec[-1][1]
            dense=expand(weight,layout,torch.float16)
            for rows in ((1,4) if args.decode_tuning or args.native_only else (1,4,129)):
                x=torch.randn(rows,width,device='cuda',dtype=torch.float16)
                expected=torch.nn.functional.linear(x,dense)
                variants={'current':lambda:linear(x,weight,layout),
                          'expanded_resident':lambda:torch.nn.functional.linear(x,dense),
                          'expand_each_call':lambda:torch.nn.functional.linear(x,expand(weight,layout,x.dtype))}
                if args.native_only:
                    from dllm.native_kquant import linear_if_enabled
                    assert os.environ.get('DLLM_KQUANT_CUDA_LIBRARY')
                    variants={'expanded_resident':variants['expanded_resident'],
                              'cuda_native':lambda:linear_if_enabled(x,weight,layout)}
                    if args.residual:
                        from experiments.kquant_residual import linear as residual_linear
                        variants['residual_int8']=lambda:residual_linear(x,weight,layout)
                elif rows<=4:
                    settings=((4,4),(8,4),(16,4),(8,8),(16,8)) if args.decode_tuning else ((1,4),(2,4),(4,4))
                    for bn,warps in settings:
                        def gemv(bn=bn,warps=warps):
                            y=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
                            _gemv[(tr.cdiv(outputs,bn),)](x,weight,weight,y,outputs,width,rows,spec,
                                bn,tr.next_power_of_2(width),False,enable_fp_fusion=False,num_warps=warps)
                            return y
                        variants[f'gemv_{bn}_w{warps}']=gemv
                    if args.decode_tuning:
                        for bn,bk in ((4,512),(8,512),(16,512),(4,1024),(8,1024)):
                            variants[f'split_{bn}_{bk}']=lambda bn=bn,bk=bk:split_gemv(x,weight,layout,bn=bn,bk=bk)
                if rows>4:
                    for bm,bn in ((16,32),(32,32),(32,64),(64,64)):
                        def gemm(bm=bm,bn=bn):
                            y=torch.empty((rows,outputs),device=x.device,dtype=x.dtype)
                            _gemm[(tr.cdiv(rows,bm),tr.cdiv(outputs,bn))](x,weight,weight,y,
                                rows,outputs,width,spec,bm,bn,256,False,
                                enable_fp_fusion=False,num_stages=2)
                            return y
                        variants[f'gemm_{bm}_{bn}_256']=gemm
                for variant,fn in variants.items():
                    ms,actual=timing(fn)
                    relative=((actual.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                    assert torch.isfinite(actual).all() and relative<.001,(name,rows,variant,relative)
                    print(json.dumps(dict(name=name,rows=rows,width=width,outputs=outputs,
                        variant=variant,ms=ms,relative_rms=relative)),flush=True)
            del weight,dense


if __name__=='__main__':main()

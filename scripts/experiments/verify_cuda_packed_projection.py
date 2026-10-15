"""Tiny shared-GPU projection gate, not expert/model/performance qualification."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('DLLM_SHARED_CUDA_GUARD')!='1':
        parser.error('requires shared guard and explicit GPU UUID')
    if args.output.exists():parser.error('new output required')
    import torch
    from dllm.cuda_packed_projection import cuda_packed_linear
    from dllm.packed_projection import packed_linear
    from tests.test_packed_projection import fixture
    from scripts.experiments.split_gguf import save
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,model_qualified=False,performance_qualified=False,device=torch.cuda.get_device_name(),
                torch=torch.__version__,cases=[])
    save(args.output,report)
    try:
        for format in ('e2m1','e4m3fn','e4m3fn_f32','q4_0_f16','q8_0_f16'):
            if format in ('e2m1','e4m3fn'):
                hidden,weight,scales,dense=fixture(format,rows=133,columns=160)
            else:
                generator=torch.Generator().manual_seed(291)
                hidden=torch.randn(3,64,generator=generator)
                if format=='e4m3fn_f32':
                    weight=torch.randint(0,119,(9,64),dtype=torch.uint8,generator=generator)
                    scales=torch.tensor([[.125]],dtype=torch.float32)
                elif format=='q4_0_f16':
                    weight=torch.randint(0,256,(9,32),dtype=torch.uint8,generator=generator)
                    scales=torch.full((9,2),.125,dtype=torch.float16)
                else:
                    weight=torch.randint(-127,128,(9,64),dtype=torch.int8,generator=generator)
                    scales=torch.full((9,2),.125,dtype=torch.float16)
                dense=None
            for dtype in (torch.float32,torch.float16,torch.bfloat16):
                source=hidden.to(dtype); x=source.cuda();w=weight.cuda();s=scales.cuda()
                expected=(source.float()@dense.T if dense is not None else
                          packed_linear(source,weight,scales,format=format,row_chunk=3,column_chunk=17))
                original=w.clone()
                actual=cuda_packed_linear(x,w,s,format=format,row_chunk=17,column_chunk=31)
                torch.testing.assert_close(actual.cpu(),expected,rtol=2e-5,atol=1e-3)
                torch.testing.assert_close(w,original,rtol=0,atol=0)
                partial=cuda_packed_linear(x,w,s,format=format,row_chunk=17,column_chunk=31,row_range=(1,3))
                torch.testing.assert_close(partial.cpu(),expected[:,1:3],rtol=2e-5,atol=1e-3)
                if cuda_packed_linear(x[:0],w,s,format=format).shape!=(0,weight.shape[0]):
                    raise AssertionError('empty batch shape')
                report['cases'].append(dict(format=format,dtype=str(dtype),shape=list(actual.shape),
                    max_absolute=float((actual.cpu()-expected).abs().max()),row_range=True,empty=True))
                save(args.output,report)
                del x,w,s,original,actual,partial
        torch.cuda.synchronize()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

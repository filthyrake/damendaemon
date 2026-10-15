"""Tiny independent-scalar expert gate; no routing or modern-model qualification."""
import argparse
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('DLLM_SHARED_CUDA_GUARD')!='1':
        parser.error('requires shared CUDA guard')
    if args.output.exists():parser.error('new report path required')
    import torch
    from dllm.cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
    from tests.test_packed_moe import expert,dense_expert
    from scripts.experiments.split_gguf import save
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],
                model_qualified=False,routing_qualified=False,performance_qualified=False)
    save(args.output,report)
    try:
        a,ad=expert('e2m1');b,bd=expert('e4m3fn')
        variants=[('e2m1',(a.gate,a.up,a.down),ad),('e4m3fn',(b.gate,b.up,b.down),bd),
                  ('mixed',(a.gate,b.up,a.down),(ad[0],bd[1],ad[2]))]
        source=torch.randn(3,32,generator=torch.Generator().manual_seed(219))*.2
        coefficients=torch.tensor([[.25],[0.],[1.]],dtype=torch.float32)
        for label,matrices,dense in variants:
            resident=[CudaPackedMatrix(m.weight.cuda(),m.scales.cuda(),m.format) for m in matrices]
            prepared=CudaPackedExpert(*resident,row_chunk=13,column_chunk=17,activation_limit=2)
            for dtype in (torch.float32,torch.float16,torch.bfloat16):
                hidden=source.to(dtype);x=hidden.cuda();c=coefficients.cuda()
                for weighted in (False,True):
                    result=prepared.forward(x,route_weights=c if weighted else None)
                    expected=dense_expert(hidden,dense,coefficients if weighted else None)
                    rtol,atol=(2e-4,1e-5) if dtype==torch.float32 else (.02,.002)
                    torch.testing.assert_close(result.cpu(),expected,rtol=rtol,atol=atol)
                    order=torch.tensor([2,0,0],device='cuda')
                    repeated=prepared.forward(x[order],route_weights=c[order] if weighted else None)
                    torch.testing.assert_close(repeated,result[order],rtol=rtol,atol=atol)
                    report['cases'].append(dict(format=label,dtype=str(dtype),weighted=weighted,
                        max_absolute=float((result.cpu().float()-expected.float()).abs().max()),reorder=True))
                    save(args.output,report)
                if prepared.forward(x[:0]).shape!=(0,32):raise AssertionError('empty expert')
                del x,c,result,repeated,order
            for original,actual in zip(matrices,resident):
                torch.testing.assert_close(original.weight,actual.weight.cpu(),rtol=0,atol=0)
                torch.testing.assert_close(original.scales,actual.scales.cpu(),rtol=0,atol=0)
            del prepared,resident
        torch.cuda.synchronize()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

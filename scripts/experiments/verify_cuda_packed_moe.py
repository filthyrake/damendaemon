"""Tiny routed CUDA composition versus independent dense CPU experts."""
import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new output path required')
    import torch
    from dllm.cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
    from dllm.cuda_packed_moe import cuda_packed_moe
    from tests.test_packed_moe import expert,dense_expert
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],
                model_qualified=False,performance_qualified=False)
    save(args.output,report)
    try:
        controls=[expert('e2m1'),expert('e4m3fn')]
        bank=[]
        for original,_ in controls:
            matrices=[CudaPackedMatrix(m.weight.cuda(),m.scales.cuda(),m.format)
                      for m in (original.gate,original.up,original.down)]
            bank.append(CudaPackedExpert(*matrices,row_chunk=13,column_chunk=17,activation_limit=2))
        source=torch.randn(3,32,generator=torch.Generator().manual_seed(225))*.2
        ids=torch.tensor([[1,1],[1,0],[0,1]],dtype=torch.int64)
        weights=torch.tensor([[.25,.5],[0.,.75],[1.,.125]],dtype=torch.float32)
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            hidden=source.to(dtype)
            for shared in (False,True):
                result=cuda_packed_moe(hidden.cuda(),bank,ids.cuda(),weights.cuda(),shared=bank[0] if shared else None)
                expected=torch.zeros_like(hidden,dtype=torch.float32)
                for row in range(3):
                    for route in range(2):
                        dense=controls[int(ids[row,route])][1]
                        expected[row:row+1]+=dense_expert(hidden[row:row+1],dense,weights[row:row+1,route:route+1]).float()
                    if shared:expected[row:row+1]+=dense_expert(hidden[row:row+1],controls[0][1],None).float()
                expected=expected.to(dtype)
                rtol,atol=(2e-4,1e-5) if dtype==torch.float32 else (.02,.002)
                torch.testing.assert_close(result.cpu(),expected,rtol=rtol,atol=atol)
                report['cases'].append(dict(dtype=str(dtype),shared=shared,
                    max_absolute=float((result.cpu().float()-expected.float()).abs().max())))
                save(args.output,report)
            empty=cuda_packed_moe(hidden[:0].cuda(),bank,ids[:0].cuda(),weights[:0].cuda(),shared=bank[0])
            if empty.shape!=(0,32):raise AssertionError('empty routed result')
        for invalid in (-1,2,2**60):
            try:cuda_packed_moe(source[:1].cuda(),bank,torch.tensor([[invalid]],device='cuda'),weights[:1,:1].cuda())
            except ValueError:pass
            else:raise AssertionError('invalid route admitted')
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

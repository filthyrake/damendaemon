"""Tiny guarded CUDA Flash sparse-layer proof against expanded CPU attention."""
import argparse
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_glm_flash_sparse import CudaFlashSparseAttention
    from dllm.cuda_glm_flash_indexer import CudaFlashIndexer
    from dllm.prepared_projection import PackedProjection
    from dllm.cuda_prepared_projection import CudaPackedProjection
    from tests.test_glm_flash_sparse import fixture,control
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except (ValueError,RuntimeError):return
        raise AssertionError('invalid input admitted')
    try:
        for packed in (False,True):
            cpu,dense,hidden=fixture(packed);expected,ids=control(hidden,dense,cpu.indexer)
            index=CudaFlashIndexer({n:v.cuda() for n,v in cpu.indexer.weights.items()},heads=2,pool_size=2,top_k=2)
            def move(w):
                return CudaPackedProjection(w.weight.cuda(),w.scales.cuda(),format=w.format,**w.options) if isinstance(w,PackedProjection) else w.cuda()
            weights={n:move(v) for n,v in cpu.weights.items()}
            gpu=CudaFlashSparseAttention(weights,index,heads=2,head_dim=3,value_dim=2)
            x=hidden.cuda();errors=[]
            for chunks in ((7,),(1,1,1,1,1,1,1),(3,2,2)):
                state=gpu.new_state(2,8);cursor=0;outputs=[];selected=[]
                for size in chunks:
                    out,chosen=gpu.forward(x[:,cursor:cursor+size],state,position=cursor)
                    outputs.append(out);selected.append(chosen);cursor+=size
                actual=torch.cat(outputs,1).cpu()
                torch.testing.assert_close(actual,expected,atol=3e-7,rtol=3e-5)
                torch.testing.assert_close(torch.cat(selected,1).cpu(),ids,atol=0,rtol=0)
                errors.append(float((actual-expected).abs().max()));state.close()
            state=gpu.new_state(2,8);gpu.forward(x[:,:3],state,position=0)
            old=[v.clone() for v in (state.latents,state.keys,state.gates)]
            with patch('dllm.glm_flash_sparse.linear',side_effect=RuntimeError('injected late projection')):
                try:gpu.forward(x[:,3:],state,position=3)
                except RuntimeError as error:assert 'injected late projection' in str(error)
                else:raise AssertionError('injected failure was ignored')
            assert state.position==3
            for a,b in zip((state.latents,state.keys,state.gates),old):torch.testing.assert_close(a,b,atol=0,rtol=0)
            order=torch.tensor([1,0,1],device='cuda');state.reorder(order)
            state.latents[0].add_(1);torch.testing.assert_close(state.latents[2],old[0][1],atol=0,rtol=0);state.latents[0].copy_(old[0][1])
            actual,chosen=gpu.forward(x[order,3:],state,position=3)
            torch.testing.assert_close(actual.cpu(),expected[[1,0,1],3:],atol=3e-7,rtol=3e-5)
            torch.testing.assert_close(chosen.cpu(),ids[[1,0,1],3:],atol=0,rtol=0)
            state.reorder(torch.tensor([2],device='cuda'));assert state.latents.shape[0]==1
            rejected(lambda:state.reorder(torch.tensor([0])))
            rejected(lambda:gpu.forward(hidden[:1,:1],state,position=7))
            rejected(lambda:gpu.new_state(2,8,max_cache_bytes=1))
            for option in ('max_workspace_bytes','max_output_bytes'):
                small=CudaFlashSparseAttention(weights,index,heads=2,head_dim=3,value_dim=2,**{option:1})
                fresh=small.new_state(2,8);rejected(lambda:small.forward(x,fresh,position=0));assert fresh.position==0;fresh.close()
            state.close();state.close();rejected(lambda:gpu.forward(x[:1,:1],state,position=7))
            assert state.latents is None and state.keys is None and state.gates is None
            report['cases'].append(dict(packed=packed,max_absolute=max(errors),ids_exact=True,chunk_decode=True,reorder=True,failure_atomic=True,closed=True))
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

"""Tiny bounded CUDA DeepSeek projected compression gate against scalar pooling."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

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
    from damendaemon.cuda_compressed_pool import pool_projected,reorder_pool
    from tests.test_compressed_pool import CompressedPoolTests,scalar_pool
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid input admitted')
    try:
        for ratio in (1,2,4,8,128):
            host=CompressedPoolTests().fixture(ratio,259 if ratio==128 else 19)
            values,gates,offsets=[x.cuda() for x in host]
            expected=scalar_pool(*host);actual,state,starts=pool_projected(values,gates,offsets)
            torch.testing.assert_close(actual.cpu(),expected,atol=1e-6,rtol=2e-5)
            torch.testing.assert_close(starts.cpu(),torch.arange(values.shape[1]//ratio)*ratio)
            for step in ((31,) if ratio==128 else (1,3,7)):
                current=None;parts=[];positions=[]
                for begin in range(0,values.shape[1],step):
                    part,current,ids=pool_projected(values[:,begin:begin+step],gates[:,begin:begin+step],offsets,state=current)
                    parts.append(part);positions.append(ids)
                torch.testing.assert_close(torch.cat(parts,1),actual,atol=0,rtol=0)
                torch.testing.assert_close(torch.cat(positions),starts)
                torch.testing.assert_close(current.values,state.values,atol=0,rtol=0)
                torch.testing.assert_close(current.scores,state.scores,atol=0,rtol=0)
            before=(state.values.clone(),state.scores.clone())
            empty,replacement,ids=pool_projected(values[:,:0],gates[:,:0],offsets,state=state)
            assert empty.shape[1]==0 and ids.numel()==0 and replacement.position==state.position
            assert replacement.values.data_ptr()!=state.values.data_ptr()
            order=torch.tensor([1,0,1],device='cuda');reordered=reorder_pool(state,order)
            torch.testing.assert_close(reordered.values,state.values[order],atol=0,rtol=0)
            reordered.values[0].add_(1);torch.testing.assert_close(reordered.values[2],state.values[1],atol=0,rtol=0)
            for option in ('max_workspace_bytes','max_output_bytes'):
                with patch('damendaemon.cuda_compressed_pool.torch.isfinite',side_effect=AssertionError('scan before admission')):
                    rejected(lambda:pool_projected(values,gates,offsets,**{option:1}))
                    rejected(lambda:reorder_pool(state,order,**{option:1}))
            rejected(lambda:pool_projected(values+float('nan'),gates,offsets,state=state))
            rejected(lambda:pool_projected(values,gates.cpu(),offsets))
            rejected(lambda:reorder_pool(state,order.cpu()))
            rejected(lambda:reorder_pool(state,order+10))
            with patch('damendaemon.cuda_compressed_pool._pool',side_effect=RuntimeError('injected core failure')):
                try:pool_projected(values,gates,offsets,state=state)
                except RuntimeError as error:assert 'injected core failure' in str(error)
                else:raise AssertionError('injected failure ignored')
            for a,b in zip((state.values,state.scores),before):torch.testing.assert_close(a,b,atol=0,rtol=0)
            report['cases'].append(dict(ratio=ratio,max_absolute=float((actual.cpu()-expected).abs().max()),chunks_exact=True,state_unchanged=True,reorder=True))
        report.update(passed=True,preallocation_bounds=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

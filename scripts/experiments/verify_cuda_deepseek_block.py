"""Guarded tiny CUDA DeepSeek blocks against decoded CPU controls."""
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
    from scripts.experiments.cuda_deepseek_fixture import prepare_block
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    try:
        for ratio,hashed in ((0,False),(4,False),(4,True)):
            candidate,control,hidden,ids=prepare_block(ratio,hashed)
            device=hidden.cuda();tokens=ids.cuda()
            expected,reference=control.prefill(hidden[:,:3],ids[:,:3],max_tokens=10,query_chunk=8)
            actual,state=candidate.prefill(device[:,:3],tokens[:,:3],max_tokens=10,query_chunk=8)
            try:
                torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                error=float((actual.cpu()-expected).abs().max())
                if hashed:
                    before=state.attention.cache.clone()
                    try:candidate.decode(device[:,3:4],state,torch.tensor([[99]],device='cuda'),position=3)
                    except ValueError:pass
                    else:raise AssertionError('invalid token admitted')
                    torch.testing.assert_close(state.attention.cache,before,rtol=0,atol=0,equal_nan=True)
                    assert not state.attention.poisoned and state.attention.position==3
                for position in range(3,7):
                    actual=candidate.decode(device[:,position:position+1],state,tokens[:,position:position+1],position=position)
                    expected=control.decode(hidden[:,position:position+1],reference,ids[:,position:position+1],position=position)
                    torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                    error=max(error,float((actual.cpu()-expected).abs().max()))
                state.reorder(torch.tensor([0,0],device='cuda'));reference.reorder(torch.tensor([0,0]))
                actual=candidate.decode(device[:,7:8].expand(2,-1,-1,-1),state,tokens[:,7:8].expand(2,-1),position=7)
                expected=control.decode(hidden[:,7:8].expand(2,-1,-1,-1),reference,ids[:,7:8].expand(2,-1),position=7)
                torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                with patch.object(candidate.backend,'ffn',side_effect=RuntimeError('injected late FFN failure')):
                    try:candidate.decode(device[:,8:9].expand(2,-1,-1,-1),state,tokens[:,8:9].expand(2,-1),position=8)
                    except RuntimeError as exc:assert 'injected late FFN failure' in str(exc)
                    else:raise AssertionError('late failure not raised')
                assert state.attention.poisoned and state.attention.position==9
                report['cases'].append(dict(ratio=ratio,hashed=hashed,max_absolute=error,reorder=True,late_failure_poisoned=True))
            finally:state.close();state.close();reference.close()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

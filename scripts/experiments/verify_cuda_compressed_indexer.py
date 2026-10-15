"""Tiny bounded CUDA DeepSeek compressed selection versus scalar oracle."""
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
    from dllm.cuda_compressed_indexer import select_compressed
    from tests.test_compressed_indexer import CompressedIndexerTests,oracle
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
        host=CompressedIndexerTests().fixture();device=[x.cuda() for x in host]
        originals=[x.clone() for x in device]
        for scaled,dot in ((True,1.),(False,.5)):
            controls=list(host)
            if not scaled:controls[2]=controls[2]*(12**.5)*dot
            expected=oracle(*controls,4,5,128)
            for qc,kc in ((1,1),(2,3),(16,64)):
                ids,scores=select_compressed(*device,ratio=4,top_k=5,offset=128,query_chunk=qc,key_chunk=kc,scale_head_weights=scaled,dot_scale=dot)
                torch.testing.assert_close(ids.cpu(),expected[0],atol=0,rtol=0)
                torch.testing.assert_close(scores.cpu(),expected[1],atol=1e-6,rtol=2e-5)
                mask=torch.isfinite(expected[1]);error=float((scores.cpu()[mask]-expected[1][mask]).abs().max())
                report['cases'].append(dict(query_chunk=qc,key_chunk=kc,scale_head_weights=scaled,dot_scale=dot,ids_exact=True,max_absolute=error))
        q,k,w,p=device
        zero=torch.zeros_like(w);ties=select_compressed(q,k,zero,p,ratio=4,top_k=5)[0]
        torch.testing.assert_close(ties.cpu(),oracle(host[0],host[1],zero.cpu(),host[3],4,5,0)[0],atol=0,rtol=0)
        poisoned=k.clone();poisoned[:,1:]=float('nan');positions=torch.full_like(p,3)
        ids,scores=select_compressed(q,poisoned,-torch.ones_like(w),positions,ratio=4,top_k=3,key_chunk=2)
        assert (ids[...,0]==0).all() and (ids[...,1:]==-1).all()
        assert torch.isfinite(scores[...,0]).all() and torch.isneginf(scores[...,1:]).all()
        rejected(lambda:select_compressed(q,poisoned,w,p,ratio=4,top_k=5))
        rejected(lambda:select_compressed(q*0+1e30,k*0+1e30,w*0+1,p,ratio=4,top_k=5))
        for option in ('max_workspace_bytes','max_output_bytes'):
            with patch('dllm.cuda_compressed_indexer.torch.isfinite',side_effect=AssertionError('scan before admission')):
                rejected(lambda:select_compressed(*device,ratio=4,top_k=5,**{option:1}))
        rejected(lambda:select_compressed(q,k,w,p.cpu(),ratio=4,top_k=5))
        rejected(lambda:select_compressed(q,k,w,p-100,ratio=4,top_k=5))
        for call,shape in ((lambda:select_compressed(q,k[:,:0],w,p,ratio=4,top_k=5),(2,5,0)),
                           (lambda:select_compressed(q[:,:0],k,w[:,:0],p[:,:0],ratio=4,top_k=5),(2,0,5))):
            ids,scores=call();assert ids.shape==scores.shape==shape
        for a,b in zip(device,originals):torch.testing.assert_close(a,b,atol=0,rtol=0)
        report.update(passed=True,ties_exact=True,future_poison_masked=True,visible_nonfinite_rejected=True,preallocation_bounds=True,inputs_unchanged=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

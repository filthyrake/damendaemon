"""Guarded logical CUDA DeepSeek ranges; no physical multi-GPU claim."""
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
    from damendaemon.cuda_deepseek_layer_stage import CudaDeepSeekLayerStage
    from damendaemon.floating_model import FloatingModel
    from scripts.experiments.cuda_deepseek_fixture import prepare_block
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],physical_multi_gpu_qualified=False)
    save(args.output,report);states=[];reference=None
    try:
        prepared=[prepare_block(ratio,hashed) for ratio,hashed in ((0,False),(4,True),(128,False))]
        rng=torch.Generator().manual_seed(258)
        embedding=torch.randn(4,8,generator=rng)*.1
        specs={'head.weight':(4,8),'norm.weight':(8,),'hc_head_fn':(2,16),'hc_head_scale':(1,),'hc_head_base':(2,)}
        head={name:torch.randn(shape,generator=rng)*.2 for name,shape in specs.items()}
        control=FloatingModel(embedding,[p[1] for p in prepared],head)
        gpu_embedding=embedding.cuda();gpu_head={k:v.cuda() for k,v in head.items()}
        tokens=torch.tensor([[0,1,2,3,1,2,3,0]]);device=tokens.cuda()
        for cuts in ((1,),(1,2)):
            bounds=(0,*cuts,3)
            stages=[CudaDeepSeekLayerStage([p[0] for p in prepared[start:end]],start=start,total=3,vocab_size=4,
                embedding=gpu_embedding if start==0 else None,head=gpu_head if end==3 else None)
                for start,end in zip(bounds,bounds[1:])]
            value=None;states=[]
            for stage in stages:
                value,state=stage.prefill(device[:,:3],hidden=value,max_tokens=9,query_chunk=8);states.append(state)
            expected,reference=control.prefill(tokens[:,:3],max_tokens=9,query_chunk=8)
            torch.testing.assert_close(value.cpu(),expected,rtol=4e-5,atol=4e-6)
            error=float((value.cpu()-expected).abs().max())
            def decode(ids,position):
                value=None
                for stage,state in zip(stages,states):value=stage.decode(ids,state,position=position,hidden=value)
                return value
            for position in (3,4):
                actual=decode(device[:,position:position+1],position)
                expected=control.decode(tokens[:,position:position+1],reference,position=position)
                torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
                torch.testing.assert_close(actual.argmax(-1).cpu(),expected.argmax(-1),rtol=0,atol=0)
                error=max(error,float((actual.cpu()-expected).abs().max()))
            for state in states:state.reorder(torch.tensor([0,0],device='cuda'))
            reference.reorder(torch.tensor([0,0]))
            actual=decode(device[:,5:6].expand(2,-1),5)
            expected=control.decode(tokens[:,5:6].expand(2,-1),reference,position=5)
            torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
            for state in states:state.reorder(torch.tensor([1],device='cuda'))
            try:stages[0].decode(torch.tensor([[4]],device='cuda'),states[0],position=6)
            except ValueError:pass
            else:raise AssertionError('invalid token admitted')
            assert all(s.position==6 and not s.poisoned for s in states)
            with patch.object(stages[-1],'_logits',side_effect=RuntimeError('injected stage head failure')):
                try:decode(device[:,6:7],6)
                except RuntimeError as exc:assert 'injected stage head failure' in str(exc)
                else:raise AssertionError('late head failure not raised')
            assert states[-1].poisoned and states[-1].position==6
            for state in states:state.close();state.close();assert state.owner is None and not state.layers
            reference.close();reference=None;states=[]
            report['cases'].append(dict(stages=len(stages),max_absolute=error,greedy_exact=True,reorder=True,owned_failure_poisoned=True))
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for state in states:state.close()
        if reference is not None:reference.close()
        save(args.output,report)


if __name__=='__main__':main()

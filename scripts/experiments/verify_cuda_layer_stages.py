"""Guarded logical CUDA GLM/Flash layer ranges; not physical multi-GPU proof."""
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
    from damendaemon.cuda_layer_stages import CudaGlmLayerStage,CudaFlashLayerStage
    from damendaemon.cuda_glm_placement import place_cuda_glm
    from damendaemon.cuda_glm_flash_placement import place_cuda_flash
    from tests.test_glm_packed_model import model_pair
    from tests.test_cuda_glm_flash_placement import fixture
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],physical_multi_gpu_qualified=False)
    save(args.output,report);states=[]
    try:
        tokens=torch.tensor([[1,2,3,4,5],[3,2,1,5,4]])
        for family in ('glm','flash'):
            if family=='glm':
                source,control=model_pair()
                source.blocks=(*source.blocks,source.blocks[-1]);control.blocks=(*control.blocks,control.blocks[-1])
                model=place_cuda_glm(source,device='cuda:0',max_bytes=1024**2).model
                stage_type=CudaGlmLayerStage;cuts=((2,),(2,3))
            else:
                source=control=fixture();model=place_cuda_flash(source,device='cuda:0',max_bytes=1024**2).model
                stage_type=CudaFlashLayerStage;cuts=((1,),(1,2))
            expected,reference=control.prefill(tokens,capacity=7,all_logits=True);reference.close()
            for boundaries in cuts:
                boundaries=(0,*boundaries,len(model.blocks));stages=[]
                for start,end in zip(boundaries,boundaries[1:]):
                    stages.append(stage_type(model.blocks[start:end],start=start,total=len(model.blocks),
                        embedding=model.embedding if start==0 else None,norm=model.norm if end==len(model.blocks) else None,
                        head=model.head if end==len(model.blocks) else None,epsilon=model.epsilon))
                states=[stage.new_state(2,7) for stage in stages]
                def forward(inputs,position):
                    value=inputs
                    for stage,state in zip(stages,states):
                        value=stage.forward(value,state,position=position,all_logits=stage.last)
                    return value
                actual=forward(tokens[:,:2].cuda(),0)
                torch.testing.assert_close(actual.cpu(),expected[:,:2],rtol=8e-5,atol=4e-6)
                error=float((actual.cpu()-expected[:,:2]).abs().max())
                order=torch.tensor([1,0,1])
                for state in states:state.reorder(order.cuda())
                for position in range(2,5):
                    actual=forward(tokens[order,position:position+1].cuda(),position)
                    target=expected[order,position:position+1]
                    torch.testing.assert_close(actual.cpu(),target,rtol=8e-5,atol=4e-6)
                    torch.testing.assert_close(actual.argmax(-1).cpu(),target.argmax(-1),rtol=0,atol=0)
                    error=max(error,float((actual.cpu()-target).abs().max()))
                before=states[0].position
                try:stages[0].forward(tokens[:1,:1],states[0],position=5)
                except ValueError:pass
                else:raise AssertionError('CPU stage boundary admitted')
                assert states[0].position==before and not states[0].poisoned
                with patch.object(stages[-1],'_logits',side_effect=RuntimeError('injected stage head failure')):
                    try:forward(torch.ones(3,1,device='cuda',dtype=torch.int64),5)
                    except RuntimeError as exc:assert 'injected stage head failure' in str(exc)
                    else:raise AssertionError('stage failure not raised')
                assert states[-1].poisoned and states[-1].position==5
                for state in states:state.close();state.close();assert state.owner is None and not state.layers
                report['cases'].append(dict(family=family,stages=len(stages),max_absolute=error,greedy_exact=True,reorder=True,owned_failure_poisoned=True))
                states=[]
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for state in states:state.close()
        save(args.output,report)


if __name__=='__main__':main()

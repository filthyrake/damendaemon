"""Guarded range-loaded CUDA stages against whole CPU models; one physical GPU."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from pathlib import Path
import tempfile
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from damendaemon.cuda_stage_placement import place_cuda_stage
    from damendaemon import glm_reference_loader,glm_flash_reference_loader,deepseek_reference_loader
    from tests import test_glm_reference_loader as glm_fixture,test_glm_flash_reference_loader as flash_fixture,test_deepseek_reference_loader as ds_fixture
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],physical_multi_gpu_qualified=False)
    save(args.output,report);states=[];reference=None
    try:
        tokens=torch.tensor([[1,2,3,4,5]])
        for family,fixture,loader,ranges in (
                ('glm',glm_fixture,glm_reference_loader,((0,2),(2,3))),
                ('flash',flash_fixture,glm_flash_reference_loader,((0,1),(1,3))),
                ('deepseek',ds_fixture,deepseek_reference_loader,((0,1),(1,3)))):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp);fixture.fixture(path);control=loader.load_reference_model(path)
                sources=[loader.load_reference_stage(path,start,end) for start,end in ranges]
                placed=[place_cuda_stage(source,device='cuda:0',max_bytes=4*1024**2) for source in sources]
                stages=[item.stage for item in placed]
                assert stages[0].head is None and stages[-1].embedding is None
                assert [(s.start,s.end) for s in stages]==list(ranges)
                states=[];value=None
                if family=='deepseek':
                    for stage in stages:
                        value,state=stage.prefill(tokens[:,:3].cuda(),hidden=value,max_tokens=6,query_chunk=8);states.append(state)
                    expected,reference=control.prefill(tokens[:,:3],max_tokens=6,query_chunk=8)
                else:
                    value=tokens[:,:3].cuda()
                    for stage in stages:
                        state=stage.new_state(1,6);states.append(state)
                        value=stage.forward(value,state,position=0,all_logits=stage.last)
                    expected_all,reference=control.prefill(tokens,capacity=6,all_logits=True)
                    expected=expected_all[:,:3]
                torch.testing.assert_close(value.cpu(),expected,rtol=8e-5,atol=4e-6)
                error=float((value.cpu()-expected).abs().max())
                for state in states:state.reorder(torch.tensor([0,0],device='cuda'))
                if family=='deepseek':reference.reorder(torch.tensor([0,0]))
                for position in (3,4):
                    ids=tokens[:,position:position+1].expand(2,-1);value=None if family=='deepseek' else ids.cuda()
                    for stage,state in zip(stages,states):
                        value=(stage.decode(ids.cuda(),state,position=position,hidden=value) if family=='deepseek'
                               else stage.forward(value,state,position=position,all_logits=stage.last))
                    target=(control.decode(ids,reference,position=position) if family=='deepseek'
                            else expected_all[:,position:position+1].expand(2,-1,-1))
                    torch.testing.assert_close(value.cpu(),target,rtol=8e-5,atol=4e-6)
                    torch.testing.assert_close(value.argmax(-1).cpu(),target.argmax(-1),rtol=0,atol=0)
                    error=max(error,float((value.cpu()-target).abs().max()))
                for state in states:state.close();assert state.owner is None and not state.layers
                reference.close();reference=None;states=[]
                report['cases'].append(dict(family=family,raw_bytes_by_stage=[item.raw_bytes for item in placed],
                    ranges=list(ranges),max_absolute=error,greedy_exact=True,reorder=True,only_owned_boundaries=True))
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for state in states:state.close()
        if reference is not None:reference.close()
        save(args.output,report)


if __name__=='__main__':main()

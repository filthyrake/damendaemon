"""Guarded independent CUDA slots versus loaded CPU slots on one physical GPU."""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report required')
    import torch
    from dllm.cuda_stage_placement import place_cuda_stage
    from dllm.cuda_stage_slots import CudaGlmStageSlots,CudaFlashStageSlots,CudaDeepSeekStageSlots
    from dllm.glm_stage_slots import GlmStageSlots
    from dllm.glm_flash_stage_slots import FlashStageSlots
    from dllm.deepseek_stage_slots import DeepSeekStageSlots
    from dllm import glm_reference_loader,glm_flash_reference_loader,deepseek_reference_loader
    from tests import test_glm_reference_loader as glm_fixture,test_glm_flash_reference_loader as flash_fixture,test_deepseek_reference_loader as ds_fixture
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],physical_multi_gpu_qualified=False)
    save(args.output,report);owned=[]
    def run(owners,tokens,starts,ids,prefill):
        tokens=tokens.to(owners[0].actual_device);hidden=owners[0].embedding(tokens)
        positions=torch.stack([torch.arange(start,start+tokens.shape[1]) for start in starts])
        for owner in owners:
            options=dict(token_ids=tokens) if isinstance(owner,DeepSeekStageSlots) else {}
            hidden=owner.forward_layers(hidden,positions,torch.tensor(ids),prefill,starts[0] if prefill else 0,**options)
        return owners[-1].output_logits(hidden[:,-1])
    try:
        for family,fixture,loader,cpu_kind,cuda_kind,ranges in (
                ('glm',glm_fixture,glm_reference_loader,GlmStageSlots,CudaGlmStageSlots,((0,2),(2,3))),
                ('flash',flash_fixture,glm_flash_reference_loader,FlashStageSlots,CudaFlashStageSlots,((0,1),(1,3))),
                ('deepseek',ds_fixture,deepseek_reference_loader,DeepSeekStageSlots,CudaDeepSeekStageSlots,((0,1),(1,3)))):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp);fixture.fixture(path)
                sources=[loader.load_reference_stage(path,start,end) for start,end in ranges]
                stages=[place_cuda_stage(source,device='cuda:0',max_bytes=4*1024**2).stage for source in sources]
                controls=[cpu_kind(source,8,3) for source in sources];owned.extend(controls)
                candidates=[cuda_kind(stage,8,3) for stage in stages];owned.extend(candidates)
                for stage,owner in zip(stages,candidates):
                    try:cpu_kind(stage,8,3)
                    except ValueError:pass
                    else:raise AssertionError('CPU slots admitted CUDA stage')
                    try:cuda_kind(stage,8,3,max_cache_bytes=owner.reserved_cache_bytes-1)
                    except ValueError:pass
                    else:raise AssertionError('aggregate reservation admitted')
                error=0.
                operations=[(torch.tensor([[1]]),[0],[2],True),
                            (torch.tensor([[2,3]]),[1],[2],True),
                            (torch.tensor([[4,3,2,1]]),[0],[0],True),
                            (torch.tensor([[0],[4]]),[4,3],[0,2],False)]
                for tokens,starts,ids,prefill in operations:
                    expected=run(controls,tokens,starts,ids,prefill);actual=run(candidates,tokens,starts,ids,prefill).cpu()
                    torch.testing.assert_close(actual,expected,rtol=8e-5,atol=4e-6)
                    torch.testing.assert_close(actual.argmax(-1),expected.argmax(-1),rtol=0,atol=0)
                    error=max(error,float((actual-expected).abs().max()))
                assert all(o.lengths==[5,0,4] for o in candidates)
                first=candidates[0];tokens=torch.tensor([[2]],device='cuda');hidden=first.embedding(tokens)
                options=dict(token_ids=tokens) if family=='deepseek' else {}
                for positions,slots in ((torch.tensor([[0]]),torch.tensor([0])),(torch.tensor([[5]],device='cuda'),torch.tensor([0]))):
                    try:first.forward_layers(hidden,positions,slots,False,**options)
                    except ValueError:pass
                    else:raise AssertionError('invalid metadata admitted')
                assert first.lengths==[5,0,4] and not first.failed
                for owner in controls+candidates:owner.release([0])
                expected=run(controls,torch.tensor([[1,2]]),[0],[0],True)
                actual=run(candidates,torch.tensor([[1,2]]),[0],[0],True).cpu()
                torch.testing.assert_close(actual,expected,rtol=8e-5,atol=4e-6)
                # Fail the second selected row after the first row has advanced.
                tokens=torch.tensor([[3],[4]],device='cuda');hidden=first.embedding(tokens)
                method='decode_activations' if family=='deepseek' else 'forward_activations'
                original=getattr(first.stage,method);calls=0
                def fail_second(*args,**kwargs):
                    nonlocal calls
                    calls+=1
                    if calls==2:raise RuntimeError('injected second row failure')
                    return original(*args,**kwargs)
                options=dict(token_ids=tokens) if family=='deepseek' else {}
                try:
                    with patch.object(first.stage,method,fail_second):
                        first.forward_layers(hidden,torch.tensor([[2],[4]]),torch.tensor([0,2]),False,**options)
                except RuntimeError as exc:
                    if str(exc)!='injected second row failure':raise
                else:raise AssertionError('injected failure absent')
                assert first.failed and first.closed and first.states==[None]*3 and first.lengths==[0]*3
                for owner in owned:owner.close()
                owned=[]
                report['cases'].append(dict(family=family,max_absolute=error,greedy_exact=True,unequal_cursors=True,
                    reordered_selection=True,chunked_continuation=True,release_reuse=True,invalid_metadata_nonmutation=True,partial_failure_closed=True))
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for owner in owned:owner.close()
        save(args.output,report)


if __name__=='__main__':main()

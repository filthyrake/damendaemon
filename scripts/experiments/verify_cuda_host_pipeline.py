"""Guarded tiny CPU/CUDA stage-local pipelines over actual loopback mutual TLS."""
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
    from dllm.cuda_host_pipeline_build import build_cuda_host_pipeline_worker
    from dllm.glm_host_pipeline_build import build_glm_host_pipeline_worker
    from dllm.glm_flash_host_pipeline_build import build_glm_flash_host_pipeline_worker
    from dllm.deepseek_host_pipeline_build import build_deepseek_host_pipeline_worker
    from dllm.cpu_snapshot_identity import fingerprint
    from dllm.host_topology import HostTopology
    from dllm import glm_reference_loader,glm_flash_reference_loader,deepseek_reference_loader
    from tests import test_glm_reference_loader as glm_fixture,test_glm_flash_reference_loader as flash_fixture,test_deepseek_reference_loader as ds_fixture
    from tests.test_cpu_host_pipeline_build import CpuHostPipelineBuildTests
    from tests.test_host_pipeline_worker import HostPipelineWorkerTests
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],physical_multi_gpu_qualified=False)
    save(args.output,report)
    helper=CpuHostPipelineBuildTests();commands=HostPipelineWorkerTests()
    try:
        for family,fixture,loader,cpu_builder,ranges,quantization in (
                ('glm',glm_fixture,glm_reference_loader,build_glm_host_pipeline_worker,((0,2),(2,3)),'fp8_reference'),
                ('flash',flash_fixture,glm_flash_reference_loader,build_glm_flash_host_pipeline_worker,((0,1),(1,3)),'fp8_reference'),
                ('deepseek',ds_fixture,deepseek_reference_loader,build_deepseek_host_pipeline_worker,((0,1),(1,3)),'fp4_fp8_reference')):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp);fixture.fixture(path);(path/'tokenizer.json').write_text('{}')
                control=loader.load_reference_model(path)
                identity=bytes.fromhex(fingerprint(path).model_id)
                def build(modes,cuts=ranges):
                    plan=HostTopology(control.config,cuts,identity,quantization=quantization,context=8,slots=2,chunk_size=8,frame_bytes=4096)
                    links=helper.links(plan);owners=[]
                    for index,(mode,group) in enumerate(zip(modes,links)):
                        options=dict(device='cuda:0',max_device_bytes=4*1024**2,max_transfer_bytes=1024**2) if mode=='cuda' else {}
                        builder=build_cuda_host_pipeline_worker if mode=='cuda' else cpu_builder
                        owner=builder(path,plan,index,group,epoch=1,verify_source=True,**options)
                        owners.append(owner);helper.addCleanup(owner.close)
                        assert owner.model_identity_verified and owner.stage.actual_device.type==mode
                    helper.together([o.handshake for o in owners]);return owners
                def expected(rows):
                    results=[]
                    for row in rows:
                        options=dict(max_tokens=8) if family=='deepseek' else dict(capacity=8)
                        logits,state=control.prefill(torch.tensor([row]),**options);state.close();results.append(logits)
                    return torch.cat(results)
                for modes in (('cpu','cuda'),('cuda','cuda'),('cuda','cpu')):
                    owners=build(modes);first=owners[0];prompts=[[1,2],[3,2,1]];error=0.
                    original=owners[-1].stage.output_logits
                    def compare(hidden):
                        nonlocal error
                        actual=original(hidden);target=expected(prompts)
                        torch.testing.assert_close(actual,target,rtol=8e-5,atol=4e-6)
                        error=max(error,float((actual-target).abs().max()));return actual
                    with patch.object(owners[-1].stage,'output_logits',compare):
                        actual=commands.command(owners,lambda:first.prefill_batch(prompts,[1,0]))
                    assert actual==expected(prompts).argmax(-1).tolist()
                    prompts=[[3,2,1,4],[1,2,5]]
                    with patch.object(owners[-1].stage,'output_logits',compare):
                        actual=commands.command(owners,lambda:first.decode([4,5],[0,1]))
                    assert actual==expected(prompts).argmax(-1).tolist()
                    commands.command(owners,lambda:first.release([0,1]))
                    commands.command(owners,lambda:first.prefill_chunk([1,2],1,0))
                    actual=commands.command(owners,lambda:first.prefill_chunk([3,4],1,2))
                    assert actual==int(expected([[1,2,3,4]]).argmax(-1))
                    commands.command(owners,lambda:first.release([1]))
                    commands.command(owners,first.stop)
                    assert all(o.closed and o.stage.stage is None and o.stage.states==[None]*2 for o in owners)
                    report['cases'].append(dict(family=family,placement=modes,max_absolute=error,greedy_exact=True,
                        verified_range_load=True,ragged_decode=True,chunk_release_reuse=True,stop_releases_weights=True))
                    helper.doCleanups()
                if family=='flash':
                    owners=build(('cuda','cuda','cuda'),((0,1),(1,2),(2,3)))
                    with patch.object(owners[1].stage,'_to_host',side_effect=RuntimeError('injected middle transfer failure')):
                        commands.command(owners,lambda:owners[0].prefill([1,2]),fails=True)
                    assert all(o.failed and o.closed and o.stage.stage is None for o in owners)
                    report['middle_transfer_failure_closed']=True;helper.doCleanups()
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        helper.doCleanups();commands.doCleanups();save(args.output,report)


if __name__=='__main__':main()

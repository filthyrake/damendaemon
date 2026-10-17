"""Small CUDA full/shared GLM cache gate against independent decoded CPU weights."""
import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from damendaemon.cuda_glm_cached_attention import CudaCachedGlmAttention
    from damendaemon.cuda_prepared_projection import CudaPackedProjection
    from damendaemon.prepared_projection import PackedProjection
    from tests.test_glm_packed_attention import models
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report);states=[]
    try:
        packed,dense,hidden=models();shared,shared_dense,_=models(True)
        def move(weight):
            if isinstance(weight,PackedProjection):return CudaPackedProjection(weight.weight.cuda(),weight.scales.cuda(),format=weight.format,**weight.options)
            return weight.cuda()
        options=dict(heads=2,nope_dim=2,rope_dim=4,value_dim=3,index_top_k=3)
        full=CudaCachedGlmAttention({n:move(w) for n,w in packed.weights.items()},**options)
        reuse=CudaCachedGlmAttention({n:move(w) for n,w in shared.weights.items()},indexer_type='shared',**options)
        states=[model.new_state(2,7) for model in (full,dense,reuse,shared_dense)]
        for start,stop in ((0,2),(2,5),(5,7)):
            x=hidden[:,start:stop]
            actual,ids=full.forward(x.cuda(),states[0],position=start)
            expected,reference_ids=dense.forward(x,states[1],position=start)
            torch.testing.assert_close(ids.cpu(),reference_ids,rtol=0,atol=0)
            torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-6)
            output,reused=reuse.forward(x.cuda(),states[2],position=start,previous_indices=ids)
            reference,_=shared_dense.forward(x,states[3],position=start,previous_indices=reference_ids)
            torch.testing.assert_close(output.cpu(),reference,rtol=3e-5,atol=3e-6)
            if reused is not ids:raise AssertionError('shared index identity changed')
            if any(s.position!=stop for s in states):raise AssertionError('cache cursor mismatch')
            report['cases'].append(dict(start=start,stop=stop,full_max_absolute=float((actual.cpu()-expected).abs().max()),shared_max_absolute=float((output.cpu()-reference).abs().max()),ids_exact=True));save(args.output,report)
            if stop==2:
                order=torch.tensor([1,0,1]);hidden=hidden[order]
                for index,state in enumerate(states):state.reorder(order.cuda() if index in (0,2) else order)
        failed=reuse.new_state(1,1);states.append(failed)
        try:reuse.forward(hidden[:1,:1].cuda(),failed,position=0,previous_indices=torch.full((1,1,1),-1,dtype=torch.int64,device='cuda'))
        except ValueError:pass
        else:raise AssertionError('invalid shared selection admitted')
        if not failed.poisoned or failed.position!=0:raise AssertionError('failed cache published cursor')
        for state in states:state.close()
        if any(not s.closed or s.owner is not None or s.latents is not None for s in states):raise AssertionError('cache cleanup incomplete')
        report.update(passed=True,reordered_slots=True,failure_poisoned=True,closed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for state in states:state.close()
        save(args.output,report)


if __name__=='__main__':main()

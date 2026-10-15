"""Tiny prepared packed CUDA GLM model logits; no official-checkpoint claim."""
import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--placement',action='store_true')
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_glm_model import CudaGlmModel,CudaGlmBlock
    from dllm.cuda_glm_cached_attention import CudaCachedGlmAttention
    from dllm.cuda_glm_ffn import CudaPackedGlmFeedForward
    from dllm.cuda_packed_placement import place_cuda_experts
    from dllm.cuda_prepared_projection import CudaPackedProjection
    from dllm.prepared_projection import PackedProjection
    from tests.test_glm_packed_model import model_pair
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],official_model_qualified=False,performance_qualified=False)
    save(args.output,report);states=[]
    try:
        cpu,control=model_pair()
        def move(weight):
            if isinstance(weight,PackedProjection):return CudaPackedProjection(weight.weight.cuda(),weight.scales.cuda(),format=weight.format,**weight.options)
            return weight.cuda()
        if args.placement:
            from dllm.cuda_glm_placement import place_cuda_glm
            placed=place_cuda_glm(cpu,device='cuda:0',max_bytes=1024*1024)
            model=placed.model
            report['placement_raw_bytes']=placed.raw_bytes
        else:
            blocks=[]
            for block in cpu.blocks:
                source=block.attention
                attention=CudaCachedGlmAttention({n:move(w) for n,w in source.weights.items()},heads=source.heads,nope_dim=source.nope_dim,rope_dim=source.rope_dim,value_dim=source.value_dim,index_top_k=source.index_top_k,indexer_type='full' if source.indexer else 'shared')
                ffn=block.feed_forward
                bank=place_cuda_experts(ffn.experts,shared=ffn.shared,device='cuda:0',max_bytes=1024*1024)
                options={} if ffn.kind=='dense' else dict(router=ffn.router.cuda(),correction_bias=ffn.correction_bias.cuda(),shared=bank.shared,top_k=ffn.top_k,route_scale=ffn.route_scale)
                prepared=CudaPackedGlmFeedForward(bank.experts[0] if ffn.kind=='dense' else bank.experts,activation_limit=ffn.activation_limit,**options)
                blocks.append(CudaGlmBlock(attention,prepared,block.input_norm.cuda(),block.post_norm.cuda(),epsilon=block.epsilon))
            model=CudaGlmModel(cpu.embedding.cuda(),blocks,cpu.norm.cuda(),cpu.head.cuda())
        tokens=torch.tensor([[1,3,2,5,4],[2,4,3,6,5]])
        expected,reference=control.prefill(tokens,capacity=5,all_logits=True);states.append(reference)
        actual,state=model.prefill(tokens[:,:3].cuda(),capacity=5,all_logits=True);states.append(state)
        torch.testing.assert_close(actual.cpu(),expected[:,:3],rtol=4e-5,atol=4e-6)
        report['cases'].append(dict(kind='prefill',max_absolute=float((actual.cpu()-expected[:,:3]).abs().max())))
        order=torch.tensor([1,0,1]);state.reorder(order.cuda());tokens=tokens[order];expected=expected[order]
        for position in (3,4):
            actual=model.forward(tokens[:,position:position+1].cuda(),state,position=position)
            torch.testing.assert_close(actual.cpu(),expected[:,position],rtol=4e-5,atol=4e-6)
            torch.testing.assert_close(actual.argmax(-1).cpu(),expected[:,position].argmax(-1),rtol=0,atol=0)
            report['cases'].append(dict(kind='decode',position=position,max_absolute=float((actual.cpu()-expected[:,position]).abs().max())));save(args.output,report)
        fresh=model.new_state(1,1);states.append(fresh)
        model.blocks[-1].post_norm.fill_(float('nan'))
        try:model.forward(torch.tensor([[1]],device='cuda'),fresh,position=0)
        except ValueError:pass
        else:raise AssertionError('nonfinite final block accepted')
        if not fresh.poisoned or fresh.position!=0 or any(not s.poisoned for s in fresh.layers):raise AssertionError('failed model published partial cursor')
        for owned in states:owned.close()
        if any(not owned.closed or owned.owner is not None or owned.layers for owned in states):raise AssertionError('model cleanup incomplete')
        report.update(passed=True,reordered_slots=True,failure_poisoned=True,closed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for owned in states:owned.close()
        save(args.output,report)


if __name__=='__main__':main()

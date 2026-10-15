"""Tiny mixed packed CUDA Flash logits/state gate, no official-model claim."""
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
    from dllm.cuda_glm_flash_model import CudaFlashModel,CudaFlashBlock,CudaPackedFlashFeedForward
    from dllm.cuda_glm_flash_primitives import CudaFlashHyperConnection
    from dllm.cuda_glm_flash_linear import CudaFlashLinearAttention
    from dllm.cuda_glm_flash_sparse import CudaFlashSparseAttention
    from dllm.cuda_glm_flash_indexer import CudaFlashIndexer
    from dllm.cuda_packed_placement import place_cuda_experts
    from dllm.cuda_prepared_projection import CudaPackedProjection
    from dllm.prepared_projection import PackedProjection
    from tests.test_glm_flash_model import fixture
    from tests.test_glm_block import ffn_weights
    from tests import test_glm_flash_ffn as ffn_test,test_glm_flash_sparse as sparse_test,test_glm_flash_linear as linear_test,test_glm_flash_hyper as hyper_test
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],official_model_qualified=False,performance_qualified=False)
    save(args.output,report);states=[]
    try:
        cpu=fixture();blocks=[];decoded_ffns=[]
        sparse,dense_sparse,_=sparse_test.fixture(True)
        cpu.blocks[1].attention=sparse
        def move(w):
            return CudaPackedProjection(w.weight.cuda(),w.scales.cuda(),format=w.format,**w.options) if isinstance(w,PackedProjection) else w.cuda()
        def hyper(h):return CudaFlashHyperConnection(h.projection.cuda(),h.base.cuda(),h.scale.cuda(),streams=h.streams)
        for i,block in enumerate(cpu.blocks):
            a=block.attention
            if i==1:
                idx=CudaFlashIndexer({n:v.cuda() for n,v in a.indexer.weights.items()},heads=2,pool_size=2,top_k=2)
                attention=CudaFlashSparseAttention({n:move(v) for n,v in a.weights.items()},idx,heads=2,head_dim=3,value_dim=2)
            else:attention=CudaFlashLinearAttention({n:v.cuda() for n,v in a.weights.items()},heads=2,head_dim=3,kernel_size=4)
            weights=block.feed_forward.weights if i==0 else ffn_weights(True)
            if i:
                weights={n:(v[...,:4,:] if n.endswith('down_proj.weight') else v if n.endswith('correction_bias') else v[...,:4]) for n,v in weights.items()}
            packed,decoded=ffn_test.packed_ffn(weights,i>0,.5);decoded_ffns.append(decoded)
            bank=place_cuda_experts(packed.experts,shared=packed.shared,device='cuda:0',max_bytes=1024*1024)
            options={} if i==0 else dict(router=packed.router.cuda(),correction_bias=packed.correction_bias.cuda(),shared=bank.shared,top_k=2)
            feed=CudaPackedFlashFeedForward(bank.experts[0] if i==0 else bank.experts,activation_limit=.5,**options)
            blocks.append(CudaFlashBlock(attention,feed,hyper(block.attention_hyper),hyper(block.ffn_hyper),block.input_norm.cuda(),block.post_norm.cuda()))
        model=CudaFlashModel(cpu.embedding.cuda(),blocks,cpu.norm.cuda(),cpu.head.cuda())
        tokens=torch.tensor([[1,3,2,5,4],[2,4,3,6,5]])
        hidden=cpu.embedding[tokens].unsqueeze(-2).expand(-1,-1,2,-1)
        # Independent CPU scalar hyper/convolution/recurrence and expanded sparse
        # attention controls; no production block/model forward is called.
        def norm(x,w):return x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-5)*w
        def restore(x,residual,post,matrix):
            return torch.stack([post[:,:,dest,None]*x+sum(matrix[:,:,src,dest,None]*residual[:,:,src] for src in range(2)) for dest in range(2)],dim=2)
        for i,block in enumerate(cpu.blocks):
            collapsed,post,matrix=hyper_test.scalar_control(hidden,block.attention_hyper)
            x=norm(collapsed,block.input_norm)
            attended=sparse_test.control(x,dense_sparse,block.attention.indexer)[0] if i==1 else linear_test.control(x,block.attention)[0]
            residual=restore(attended,hidden,post,matrix)
            collapsed,post,matrix=hyper_test.scalar_control(residual,block.ffn_hyper)
            x=norm(collapsed,block.post_norm)
            fed=torch.stack([ffn_test.control(row,decoded_ffns[i],i>0,.5) for row in x.reshape(-1,4)]).reshape_as(x)
            hidden=restore(fed,residual,post,matrix)
        expected=norm((hidden[:,:,0]+hidden[:,:,1])/2,cpu.norm)@cpu.head.T
        actual,state=model.prefill(tokens[:,:3].cuda(),capacity=5,all_logits=True);states.append(state)
        torch.testing.assert_close(actual.cpu(),expected[:,:3],atol=3e-6,rtol=8e-5)
        report['cases'].append(dict(kind='prefill',max_absolute=float((actual.cpu()-expected[:,:3]).abs().max())))
        order=torch.tensor([1,0,1]);state.reorder(order.cuda())
        for pos in (3,4):
            actual=model.forward(tokens[order,pos:pos+1].cuda(),state,position=pos)
            target=expected[order,pos]
            torch.testing.assert_close(actual.cpu(),target,atol=3e-6,rtol=8e-5)
            torch.testing.assert_close(actual.argmax(-1).cpu(),target.argmax(-1),atol=0,rtol=0)
            report['cases'].append(dict(kind='decode',position=pos,max_absolute=float((actual.cpu()-target).abs().max())))
        actual,features=model.prefill_embeddings(cpu.embedding[tokens].cuda(),capacity=5,all_logits=True);states.append(features)
        torch.testing.assert_close(actual.cpu(),expected,atol=3e-6,rtol=8e-5)
        fresh=model.new_state(2,5);states.append(fresh)
        with patch.object(model,'_logits',side_effect=RuntimeError('injected final head failure')):
            try:model.forward(tokens[:,:1].cuda(),fresh,position=0)
            except RuntimeError as error:assert 'injected final head failure' in str(error)
            else:raise AssertionError('injected failure ignored')
        assert fresh.poisoned and fresh.position==0 and all(s.poisoned for s in fresh.layers)
        for s in states:s.close()
        assert all(s.closed and s.owner is None and not s.layers for s in states)
        report.update(passed=True,prepared_features=True,reordered_slots=True,failure_poisoned=True,closed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        for s in states:s.close()
        save(args.output,report)


if __name__=='__main__':main()

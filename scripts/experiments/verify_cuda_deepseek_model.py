"""Guarded tiny prepared CUDA DeepSeek model against decoded CPU composition."""
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
    from dllm.cuda_deepseek_model import CudaDeepSeekModel
    from dllm.floating_model import FloatingModel
    from scripts.experiments.cuda_deepseek_fixture import prepare_block
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],official_model_qualified=False)
    save(args.output,report)
    state=reference=None
    try:
        prepared=[prepare_block(ratio,hashed) for ratio,hashed in ((0,False),(4,True),(128,False))]
        rng=torch.Generator().manual_seed(251)
        embedding=torch.randn(4,8,generator=rng)*.1
        specs={'head.weight':(4,8),'norm.weight':(8,),'hc_head_fn':(2,16),'hc_head_scale':(1,),'hc_head_base':(2,)}
        head={name:torch.randn(shape,generator=rng)*.2 for name,shape in specs.items()}
        cpu=FloatingModel(embedding,[p[1] for p in prepared],head)
        gpu=CudaDeepSeekModel(embedding.cuda(),[p[0] for p in prepared],{k:v.cuda() for k,v in head.items()})
        ids=torch.tensor([[0,1,2,3,1,2,3,0,1]]);tokens=ids.cuda()
        with patch.object(gpu,'_embed',side_effect=AssertionError('embedding before admission')):
            try:gpu.prefill(tokens[:,:3],max_tokens=10,max_cache_bytes=1)
            except ValueError:pass
            else:raise AssertionError('cache budget admitted')
        expected,reference=cpu.prefill(ids[:,:3],max_tokens=10,query_chunk=8)
        actual,state=gpu.prefill(tokens[:,:3],max_tokens=10,query_chunk=8)
        def compare(actual,expected,position):
            torch.testing.assert_close(actual.cpu(),expected,rtol=4e-5,atol=4e-6)
            torch.testing.assert_close(actual.argmax(-1).cpu(),expected.argmax(-1),rtol=0,atol=0)
            report['cases'].append(dict(position=position,max_absolute=float((actual.cpu()-expected).abs().max()),greedy_exact=True))
        compare(actual,expected,3)
        try:gpu.decode(torch.tensor([[4]],device='cuda'),state,position=3)
        except ValueError:pass
        else:raise AssertionError('invalid token admitted')
        assert state.position==3 and not state.poisoned and all(s.attention.position==3 for s in state.layers)
        for position in range(3,7):
            actual=gpu.decode(tokens[:,position:position+1],state,position=position)
            expected=cpu.decode(ids[:,position:position+1],reference,position=position)
            compare(actual,expected,position+1)
        state.reorder(torch.tensor([0,0],device='cuda'));reference.reorder(torch.tensor([0,0]))
        actual=gpu.decode(tokens[:,7:8].expand(2,-1),state,position=7)
        expected=cpu.decode(ids[:,7:8].expand(2,-1),reference,position=7)
        compare(actual,expected,8)
        state.reorder(torch.tensor([1],device='cuda'))
        with patch.object(gpu,'_logits',side_effect=RuntimeError('injected late head failure')):
            try:gpu.decode(tokens[:,8:9],state,position=8)
            except RuntimeError as exc:assert 'injected late head failure' in str(exc)
            else:raise AssertionError('late failure not raised')
        assert state.poisoned and state.position==8 and all(s.attention.poisoned and s.attention.position==9 for s in state.layers)
        state.close();state.close();assert state.owner is None and not state.layers
        report.update(passed=True,reorder=True,late_failure_poisoned_all_layers=True,preallocation_admission=True,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        if state is not None:state.close()
        if reference is not None:reference.close()
        save(args.output,report)


if __name__=='__main__':main()

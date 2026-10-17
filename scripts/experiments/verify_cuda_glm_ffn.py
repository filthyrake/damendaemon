"""Bounded explicit CUDA GLM FFN versus independently decoded dense CPU control."""
import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from damendaemon.cuda_packed_placement import place_cuda_experts
    from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward,_select
    from damendaemon.glm_ffn import select_glm_experts
    from tests.test_glm_packed_model import ffns
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    try:
        hidden=torch.linspace(-1,1,72).reshape(2,6,6)[:,::2]
        for sparse,policy in ((False,'normal'),(False,'clipped'),(True,'normal'),(True,'clipped'),(True,'underflow')):
            cpu,dense=ffns(sparse)
            limit=.01 if policy=='clipped' else None
            dense.activation_limit=limit
            for expert in cpu.experts+(() if cpu.shared is None else (cpu.shared,)):
                expert.activation_limit=limit
            placed=place_cuda_experts(cpu.experts,shared=cpu.shared,device='cuda:0',max_bytes=1024*1024)
            source=torch.ones_like(hidden) if policy=='underflow' else hidden
            if policy=='underflow':cpu.router.fill_(-1000.)
            options=dict(activation_limit=limit)
            if sparse:options.update(router=cpu.router.cuda(),correction_bias=cpu.correction_bias.cuda(),shared=placed.shared,top_k=2)
            candidate=CudaPackedGlmFeedForward(placed.experts if sparse else placed.experts[0],**options)
            actual=candidate(source.cuda()).cpu();expected=dense(source)
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
            report['cases'].append(dict(sparse=sparse,policy=policy,max_absolute=float((actual-expected).abs().max()),raw_bytes=placed.raw_bytes))
            save(args.output,report)
        logits=torch.tensor([[-1000.,-1000.,-1000.],[-46.,-47.,-48.],[1.,1.,1.]])
        bias=torch.tensor([1.,0.,0.])
        weights,ids=_select(logits.cuda(),bias.cuda(),2,2.5)
        expected,expected_ids=select_glm_experts(logits,bias,2)
        torch.testing.assert_close(ids.cpu(),expected_ids,rtol=0,atol=0)
        torch.testing.assert_close(weights.cpu(),expected,rtol=1e-5,atol=1e-7)
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved(),routing_edge_cases=True)
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

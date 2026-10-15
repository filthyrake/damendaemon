"""Tiny CUDA compressor projection/pool/norm/RoPE and explicit cache simulations."""
import argparse
import math
from pathlib import Path
from unittest.mock import patch
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from dllm.cuda_floating_compressor import compress_hidden
    from dllm.deepseek_position import frequencies
    from dllm.activation_simulation import simulate_fp8,simulate_fp4,normalized_hadamard
    from tests.test_compressed_pool import scalar_pool
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid input admitted')
    try:
        for ratio in (4,8):
            for transform,width in ((None,4),('kv_fp8',66),('index_fp4',32)):
                rng=torch.Generator().manual_seed(97);channels=width*(2 if ratio==4 else 1)
                host=[torch.randn(2,19,5,generator=rng),torch.randn(channels,5,generator=rng),torch.randn(channels,5,generator=rng),
                      torch.randn(ratio,channels,generator=rng),torch.randn(width,generator=rng),frequencies(2,base=160000)]
                hidden,value,gate,offsets,norm,freq=host
                pooled=scalar_pool(hidden@value.T,hidden@gate.T,offsets)
                expected=norm*pooled/torch.sqrt(pooled.square().mean(-1,keepdim=True)+1e-6)
                for b in range(2):
                    for token in range(expected.shape[1]):
                        a,c=expected[b,token,-2:].tolist();angle=token*ratio*float(freq[0])
                        expected[b,token,-2]=a*math.cos(angle)-c*math.sin(angle)
                        expected[b,token,-1]=c*math.cos(angle)+a*math.sin(angle)
                if transform=='kv_fp8':expected[...,:-2]=simulate_fp8(expected[...,:-2])
                elif transform=='index_fp4':expected=simulate_fp4(normalized_hadamard(expected))
                device=[x.cuda() for x in host];actual,state,starts=compress_hidden(*device,cache_transform=transform)
                torch.testing.assert_close(actual.cpu(),expected,atol=2e-6,rtol=3e-5)
                torch.testing.assert_close(starts.cpu(),torch.arange(expected.shape[1])*ratio)
                current=None;parts=[]
                for begin in range(0,19,3):
                    out,current,_=compress_hidden(device[0][:,begin:begin+3],*device[1:],state=current,cache_transform=transform);parts.append(out)
                torch.testing.assert_close(torch.cat(parts,1),actual,atol=2e-6,rtol=3e-5)
                _,prior,_=compress_hidden(device[0][:,:3],*device[1:]);old=(prior.values.clone(),prior.scores.clone())
                with patch('dllm.floating_compressor.rotate_pairs',side_effect=RuntimeError('injected rotation failure')):
                    try:compress_hidden(device[0][:,3:11],*device[1:],state=prior)
                    except RuntimeError as error:assert 'injected rotation failure' in str(error)
                    else:raise AssertionError('injected failure ignored')
                for a,b in zip((prior.values,prior.scores),old):torch.testing.assert_close(a,b,atol=0,rtol=0)
                assert prior.position==3
                for option in ('max_workspace_bytes','max_output_bytes'):
                    with patch('dllm.floating_compressor.F.linear',side_effect=AssertionError('projected before budget')):
                        rejected(lambda:compress_hidden(*device,**{option:1}))
                rejected(lambda:compress_hidden(device[0],host[1],*device[2:]))
                # Finite projected/pool values whose squared variance overflows.
                bad=list(device);bad[0]=torch.ones_like(bad[0]);bad[1]=torch.full_like(bad[1],1e20);bad[2]=torch.zeros_like(bad[2])
                rejected(lambda:compress_hidden(*bad))
                report['cases'].append(dict(ratio=ratio,cache_transform=transform,max_absolute=float((actual.cpu()-expected).abs().max()),chunk_continuation=True,failure_atomic=True))
        report.update(passed=True,preallocation_bounds=True,variance_overflow_rejected=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

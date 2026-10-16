"""Bounded CUDA Flash linear layer gate; synthetic equations, not model proof."""
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
    from dllm.cuda_glm_flash_linear import CudaFlashLinearAttention
    from dllm.glm_flash_linear import FlashLinearAttention
    from tests.test_glm_flash_linear import layer,control
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except (ValueError,RuntimeError):return
        raise AssertionError('invalid input admitted')
    try:
        for kernel in (1,4):
            cpu=layer(kernel);weights={k:v.cuda() for k,v in cpu.weights.items()}
            gpu=CudaFlashLinearAttention(weights,heads=2,head_dim=3,kernel_size=kernel)
            host=torch.randn(2,5,4);hidden=host.cuda();expected,expected_state=control(host,cpu)
            whole=gpu.new_state(2,8);actual=gpu.forward(hidden,whole,position=0)
            torch.testing.assert_close(actual.cpu(),expected,atol=3e-7,rtol=3e-5)
            torch.testing.assert_close(whole.recurrent.cpu(),expected_state,atol=2e-7,rtol=3e-6)
            for chunks in ((2,2,1),(1,1,1,1,1)):
                state=gpu.new_state(2,8);cursor=0;outputs=[]
                for size in chunks:
                    outputs.append(gpu.forward(hidden[:,cursor:cursor+size],state,position=cursor));cursor+=size
                torch.testing.assert_close(torch.cat(outputs,1),actual,atol=3e-7,rtol=3e-5)
                torch.testing.assert_close(state.recurrent,whole.recurrent,atol=2e-7,rtol=3e-6)
                torch.testing.assert_close(state.convolution,whole.convolution)
                state.close()
            state=gpu.new_state(2,8);gpu.forward(hidden[:,:3],state,position=0)
            old=(state.convolution.clone(),state.recurrent.clone())
            with patch('dllm.glm_flash_linear.recurrent_attention',side_effect=RuntimeError('injected recurrence failure')):
                rejected(lambda:gpu.forward(hidden[:,3:4],state,position=3))
            assert state.position==3
            torch.testing.assert_close(state.convolution,old[0],rtol=0,atol=0)
            torch.testing.assert_close(state.recurrent,old[1],rtol=0,atol=0)
            order=torch.tensor([1,0,1],device='cuda');state.reorder(order)
            state.recurrent[0].add_(1)
            torch.testing.assert_close(state.recurrent[2],old[1][1],rtol=0,atol=0)
            state.recurrent[0].copy_(old[1][1])
            out=gpu.forward(hidden[order,3:],state,position=3)
            torch.testing.assert_close(out.cpu(),expected[[1,0,1],3:],atol=3e-7,rtol=3e-5)
            state.reorder(torch.tensor([2],device='cuda'));assert state.recurrent.shape[0]==1
            rejected(lambda:state.reorder(torch.tensor([0])))
            rejected(lambda:gpu.forward(host[:1,:1],state,position=5))
            rejected(lambda:gpu.new_state(2,8,max_cache_bytes=1))
            rejected(lambda:FlashLinearAttention(weights,heads=2,head_dim=3,kernel_size=kernel))
            for option in ('max_workspace_bytes','max_output_bytes'):
                small=CudaFlashLinearAttention(weights,heads=2,head_dim=3,kernel_size=kernel,**{option:1})
                fresh=small.new_state(2,8);rejected(lambda:small.forward(hidden,fresh,position=0));assert fresh.position==0;fresh.close()
            state.close();state.close();rejected(lambda:gpu.forward(hidden[:1,:1],state,position=5))
            assert state.recurrent is None and state.convolution is None
            whole.close()
            report['cases'].append(dict(kernel=kernel,max_absolute=float((actual.cpu()-expected).abs().max()),chunk_decode=True,reorder=True,failure_atomic=True,closed=True))
        report.update(passed=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

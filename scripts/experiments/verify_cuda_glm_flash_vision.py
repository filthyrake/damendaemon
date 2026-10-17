"""Guarded tiny CUDA Flash vision against independent CPU convolution/attention."""
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
    from damendaemon.cuda_glm_flash_vision import CudaFlashVision
    from tests.test_glm_flash_vision import fixture,control
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],official_vision_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid vision input admitted')
    try:
        config,weights=fixture();device_weights={k:v.cuda() for k,v in weights.items()}
        encoder=CudaFlashVision(config,device_weights)
        for grids in ([(1,2,4)],[(2,2,4),(1,2,2)],[(1,4,4)]):
            count=sum(t*h*w for t,h,w in grids);hidden=torch.randn(count,24);gpu=hidden.cuda();start=0;expected=[]
            for frames,height,width in grids:
                for _ in range(frames):
                    end=start+height*width;expected.append(control(hidden[start:end],config,weights,(1,height,width)));start=end
            expected=torch.cat(expected);actual=encoder.forward(gpu,grids).cpu()
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(gpu.cpu(),hidden,rtol=0,atol=0)
            if grids[0][0]>1:
                changed=gpu.clone();changed[8:]=100
                torch.testing.assert_close(encoder.forward(changed,grids)[:2].cpu(),actual[:2],rtol=0,atol=0)
            report['cases'].append(dict(grids=grids,max_absolute=float((actual-expected).abs().max())))
        for option in ('max_workspace_bytes','max_output_bytes','max_tokens'):
            bounded=CudaFlashVision(config,device_weights,**{option:1})
            with patch('damendaemon.glm_flash_vision.F.linear',side_effect=AssertionError('projection before admission')):
                rejected(lambda:bounded.forward(gpu,grids))
        invalid=gpu.clone();invalid[0,0]=float('nan');rejected(lambda:encoder.forward(invalid,grids))
        saved=device_weights['patch_embed.proj.weight'].clone()
        device_weights['patch_embed.proj.weight'].fill_(1e30)
        rejected(lambda:encoder.forward(torch.ones_like(gpu),grids))
        device_weights['patch_embed.proj.weight'].copy_(saved)
        torch.testing.assert_close(encoder.forward(gpu,grids).cpu(),actual,rtol=0,atol=0)
        for name,value in weights.items():torch.testing.assert_close(device_weights[name].cpu(),value,rtol=0,atol=0)
        report.update(passed=True,frame_isolation=True,preallocation_bounds=True,stateless_failure_recovery=True,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()

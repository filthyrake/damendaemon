"""Real-device numerical/ABBA primitive probe; not a model-serving benchmark."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from damendaemon.runtime import require_gpu
from damendaemon.quantized_kernels import expand
from scripts.experiments.hip_q8_bf16x3 import linear


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(318)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  input_output_dtype='float32', storage='q8_0', candidate_products='BF16 high/high + high/low + low/high; FP32 accumulation',
                  control='FP32 dequantization plus torch linear, TF32 disabled', cases=[])
    try:
        for m,n,k in ((5,48,5120),(31,14336,5120),(127,34816,5120),(127,5120,17408)):
            x = torch.randn((m,k), device='cuda', dtype=torch.float32)
            w = torch.randint(-127,128,(n,k), device='cuda', dtype=torch.int8)
            s = (torch.rand((n,k//32),device='cuda')*.001).half()
            def control():return torch.nn.functional.linear(x, expand(w,s,torch.float32))
            def candidate():return linear(x,w,s)
            expected, actual = control(), candidate()
            delta = actual-expected
            rms = (delta.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-6)).item()
            maximum = delta.abs().max().item()
            passed = bool(torch.isfinite(actual).all()) and rms < 1e-4 and maximum < .01
            case = dict(shape=[m,n,k],relative_rms=rms,max_absolute=maximum,passed=passed,samples=[])
            report['cases'].append(case)
            if not passed:raise AssertionError('primitive numerical gate failed')
            for _ in range(2):control();candidate()
            torch.cuda.synchronize()
            for cycle in range(5):
                for label, operation in (('control',control),('candidate',candidate),('candidate',candidate),('control',control)):
                    torch.cuda.synchronize();start=time.perf_counter()
                    result=operation();torch.cuda.synchronize()
                    case['samples'].append(dict(cycle=cycle,path=label,seconds=time.perf_counter()-start))
                    del result
            case['median_seconds']={label:statistics.median(v['seconds'] for v in case['samples'] if v['path']==label) for label in ('control','candidate')}
            del x,w,s,expected,actual,delta
            torch.cuda.empty_cache()
        report['passed']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':main()

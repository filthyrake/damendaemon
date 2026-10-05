"""Independent upstream Q4_0 byte/layout gate for grouped expert projections."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import triton.testing
from gguf.quants import quantize, dequantize
from gguf.constants import GGMLQuantizationType
from dllm.runtime import require_gpu
from experiments.moe_q4_grouped import linear
from dllm.grouped_moe import prepare_routes


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(429)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, device=torch.cuda.get_device_name(), cases=[],
                  scope='isolated grouped Q4 expert projection; upstream gguf quantization/dequantization; no runtime registration',
                  source_sha256=hashlib.sha256(Path('scripts/experiments/moe_q4_grouped.py').read_bytes()).hexdigest())
    try:
        for experts, outputs, width, top in ((5,67,96,3),(64,128,2048,8),(4,2048,2048,2)):
            original = (torch.randn(experts*outputs,width)*.1).half().float().numpy()
            packed = quantize(original,GGMLQuantizationType.Q4_0)
            blocks = packed.reshape(experts*outputs,width//32,18)
            weights = torch.from_numpy(blocks[:,:,2:].copy().reshape(experts*outputs,width//2)).cuda()
            scales = torch.from_numpy(blocks[:,:,:2].copy().view(np.float16).reshape(experts*outputs,width//32)).cuda()
            reference = torch.from_numpy(dequantize(packed,GGMLQuantizationType.Q4_0)).cuda().view(experts,outputs,width)
            for batch in (1,4,17,65):
                for concentrated in (False,True):
                    selected = (torch.zeros((batch,top),device='cuda',dtype=torch.long) if concentrated
                                else torch.randint(experts,(batch,top),device='cuda'))
                    routes = prepare_routes(selected,experts)
                    ids = selected.cpu().reshape(-1).tolist()
                    for shared in (False,True):
                        x = torch.randn(batch if shared else batch*top,width,device='cuda')
                        expected = torch.stack([torch.nn.functional.linear(x[i//top if shared else i],reference[expert])
                                                for i,expert in enumerate(ids)])
                        actual = linear(x,weights,scales,routes,outputs,shared)
                        error = actual-expected
                        rms = (error.square().mean()/expected.square().mean().clamp_min(1e-12)).sqrt().item()
                        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
                        if not bool(torch.isfinite(actual).all()) or rms>=1e-5:
                            raise AssertionError('Q4 expert numerical gate failed')
                        case = dict(experts=experts,outputs=outputs,width=width,top=top,batch=batch,
                                    concentrated=concentrated,shared=shared,relative_rms=rms,
                                    max_absolute=error.abs().max().item(),
                                    graph_ms=triton.testing.do_bench_cudagraph(
                                        lambda:linear(x,weights,scales,routes,outputs,shared),rep=50))
                        report['cases'].append(case)
                        print(json.dumps(case),flush=True)
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Independent synthetic gate for device-counted expert projection experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import triton.testing
from damendaemon.quantization import quantize, dequantize
from damendaemon.runtime import require_gpu
from experiments.moe_grouped import prepare_routes, linear


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--block-rows', type=int, choices=(16, 32), default=16)
    parser.add_argument('--block-columns', type=int, choices=(64, 128), default=64)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(342)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, block_rows=args.block_rows, block_columns=args.block_columns, cases=[], device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(Path('scripts/experiments/moe_grouped.py').read_bytes()).hexdigest(),
        scope='synthetic grouped projections; independent FP32 dense reference; not runtime registration')
    try:
        for experts, rows, top, outputs, width in ((5,1,3,67,96),(5,17,3,67,96),
                (64,65,8,128,128),(4,33,2,2048,2048)):
            for concentrated in (False, True):
                selected = (torch.zeros((rows,top),device='cuda',dtype=torch.long) if concentrated else
                            torch.randint(experts,(rows,top),device='cuda'))
                # Repeated routes, empty experts, and nonmultiple row/output tiles.
                routes = prepare_routes(selected,experts,args.block_rows)
                expected_counts = torch.bincount(selected.cpu().reshape(-1),minlength=experts)
                torch.testing.assert_close(routes.counts.cpu().long(),expected_counts)
                original = torch.randn(experts*outputs,width).half()*.1
                for quantized in (False, True):
                    if quantized:
                        weights,scales=quantize(original)
                        reference=dequantize(weights,scales).cuda().view(experts,outputs,width)
                        weights,scales=weights.cuda(),scales.cuda()
                    else:
                        weights,scales=original.cuda(),None
                        reference=weights.float().view(experts,outputs,width)
                    expert_ids=selected.cpu().reshape(-1).tolist()
                    for shared in (False,True):
                        x=torch.randn(rows if shared else rows*top,width,device='cuda')
                        expected=torch.stack([torch.nn.functional.linear(x[i//top if shared else i],reference[e])
                                              for i,e in enumerate(expert_ids)])
                        actual=linear(x,weights,scales,routes,outputs,shared,args.block_columns)
                        error=actual-expected
                        rms=(error.square().mean()/expected.square().mean().clamp_min(1e-12)).sqrt().item()
                        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=.002)
                        if not bool(torch.isfinite(actual).all()) or rms>=1e-4:
                            raise AssertionError('grouped projection numerical gate failed')
                        case=dict(experts=experts,rows=rows,top=top,outputs=outputs,width=width,
                                  concentrated=concentrated,quantized=quantized,shared=shared,
                                  relative_rms=rms,max_absolute=error.abs().max().item())
                        report['cases'].append(case)
                        case['graph_ms']=triton.testing.do_bench_cudagraph(
                            lambda:linear(x,weights,scales,routes,outputs,shared,args.block_columns),rep=50)
                        print(json.dumps(case),flush=True)
        report['passed']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

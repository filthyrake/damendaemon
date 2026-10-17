"""Native Q4 Metal against independent unpacking and CPU FP64 products."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from damendaemon import quantized_metal_q4


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(1321)
    torch.set_num_threads(4)
    report = dict(passed=False, final_report=False, checks=[], torch=str(torch.__version__),
                  source_sha256=hashlib.sha256(Path(quantized_metal_q4.__file__).read_bytes()).hexdigest(),
                  scope='native Metal Q4 projection correctness; no model or performance qualification')
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU required')
        for dtype in (torch.float32, torch.float16):
            for rows, outputs, width in ((1,7,32),(2,31,96),(3,65,256),(4,129,768),
                                         (5,33,64),(8,65,1536),(16,1024,4096),(17,65,1536),(31,13,128)):
                for biased in (False, True):
                    packed = torch.randint(0, 256, (outputs, width//2), dtype=torch.uint8)
                    scales = ((torch.rand(outputs, width//32)-.5)*.06).half()
                    scales[::3,0] = 0
                    x = (torch.randn(rows,width)*(3 if dtype==torch.float16 else 1.e4)).to(dtype)
                    if dtype==torch.float32:x[:,-1] = 1.e5
                    bias = torch.randn(outputs).to(dtype) if biased else None
                    blocks = packed.reshape(outputs, width//32, 16).long()
                    integers = torch.cat((blocks & 15, blocks >> 4), dim=-1)-8
                    dense = (integers.double()*scales.double()[:,:,None]).reshape(outputs,width)
                    if dtype==torch.float16:dense = dense.half().double()
                    expected = F.linear(x.double(),dense,None if bias is None else bias.double())
                    actual = quantized_metal_q4.linear(x.to('mps'),packed.to('mps'),scales.to('mps'),
                        None if bias is None else bias.to('mps')).cpu().double()
                    error = actual-expected
                    relative = (error.square().mean()/expected.square().mean().clamp_min(1.e-12)).sqrt().item()
                    passed = bool(torch.isfinite(actual).all()) and relative < (.0005 if dtype==torch.float16 else .00001)
                    row = dict(dtype=str(dtype),rows=rows,outputs=outputs,width=width,bias=biased,
                               relative_rms=relative,max_absolute=error.abs().max().item(),passed=passed)
                    report['checks'].append(row)
                    if not passed:raise AssertionError(row)
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['final_report'] = True
        output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(report['checks']))))


if __name__ == '__main__':main()

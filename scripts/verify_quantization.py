"""CPU Q8_0 format comparison and optional native CUDA matrix gate."""
import argparse
import json
import sys
import torch
from dllm.quantization import quantize, dequantize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gguf-source', help='read-only upstream gguf-py source for independent format comparison')
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(11)
    checks = []
    if args.gguf_source:
        sys.dont_write_bytecode = True
        sys.path.insert(0,args.gguf_source)
        import numpy as np
        from gguf.quants import quantize as reference_quantize
        from gguf.constants import GGMLQuantizationType
        for weight in (torch.randn(71,96), torch.zeros(3,64),
                       torch.tensor([[-127,-63.5,-.5,0,.5,63.5,127,1]*4],dtype=torch.float32),
                       torch.randn(19,128).half().float() * 1e-5):
            data, scales = quantize(weight)
            packed = np.concatenate((scales.numpy().reshape(-1,1).view(np.uint8),
                                     data.numpy().reshape(-1,32).view(np.uint8)),axis=1)
            expected = reference_quantize(weight.numpy(), GGMLQuantizationType.Q8_0).reshape(-1,34)
            assert np.array_equal(packed,expected)
            checks.append({'kind':'format', 'shape':list(weight.shape), 'bit_exact':True})
    if args.cuda:
        from dllm.runtime import require_gpu
        from dllm.quantized_kernels import linear
        require_gpu()
        for rows,width,outputs in ((1,896,151936),(2,3072,32064),(4,8192,3072),(3,96,113),(7,96,113),
                                   (17,96,113),(23,4096,3072),(33,896,1024),(64,4096,3072),
                                   (65,96,113),(92,14336,4096),(128,96,113),(129,96,113)):
            weight = torch.randn(outputs,width) * .02
            data, scales = quantize(weight)
            data, scales = data.cuda(), scales.cuda()
            for dtype in (torch.float16,torch.float32):
                x = torch.randn(rows,width,device='cuda',dtype=dtype)
                bias = torch.randn(outputs,device='cuda',dtype=dtype)
                expected = torch.nn.functional.linear(x,dequantize(data,scales,dtype),bias)
                actual = linear(x,data,scales,bias)
                error = (actual.float()-expected.float())
                relative = (error.square().mean()/expected.float().square().mean()).sqrt().item()
                assert torch.isfinite(actual).all() and relative < .001
                checks.append({'kind':'cuda', 'rows':rows,'width':width,'outputs':outputs,
                               'dtype':str(dtype),'relative_rms':relative})
    if not checks:
        p.error('choose an independent format or CUDA check')
    print(json.dumps({'passed':True,'checks':checks},indent=2))


if __name__ == '__main__':
    main()

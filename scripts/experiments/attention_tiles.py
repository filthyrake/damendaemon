"""Isolated fixed-cache Qwen3 decode attention tile experiment on A2."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.testing
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from dllm.kernels import _attention, _merge
from dllm.runtime import require_gpu


def attention(q, k, v, positions, slots, block):
    batch, heads, _, dim = q.shape
    _, kv_heads, context, _ = k.shape
    splits = triton.cdiv(context, block)
    out = torch.empty_like(q)
    partial = torch.empty((batch*heads, splits, dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch*heads, splits), device=q.device, dtype=torch.float32)
    _attention[(batch*heads, splits)](
        q, k, v, positions, slots, out, partial, lse,
        positions, False, positions, False, positions, 0, 1.,
        positions, False, context, 0, heads, kv_heads, dim, context,
        splits, block, True, 0, dim**-.5, triton.next_power_of_2(dim), num_warps=4)
    _merge[(batch*heads,)](partial, lse, out, positions, False, heads, dim,
        splits, triton.next_power_of_2(splits), triton.next_power_of_2(dim))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise RuntimeError('this tuning experiment is limited to A2')
    torch.set_num_threads(4)
    torch.manual_seed(781)
    report = dict(passed=False, device=torch.cuda.get_device_name(), cases=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='fixed-cache Qwen3 geometry only; independent FP32 math SDPA; no runtime policy change')
    try:
        with torch.inference_mode():
            for batch in (1, 4):
                for position in (0, 129, 255, 256, 511, 2047):
                    q = torch.randn(batch, 32, 1, 128, device='cuda', dtype=torch.float16)
                    k = torch.randn(batch, 8, 2048, 128, device='cuda', dtype=torch.float16)
                    v = torch.randn_like(k)
                    slots = torch.arange(batch-1, -1, -1, device='cuda')
                    positions = torch.full((batch,), position, device='cuda', dtype=torch.long)
                    # Invalid future cache must not affect output, including split boundaries.
                    k[:, :, position+1:] = float('nan')
                    v[:, :, position+1:] = float('nan')
                    with sdpa_kernel(SDPBackend.MATH):
                        expected = F.scaled_dot_product_attention(q.float(),
                            k[slots, :, :position+1].float(), v[slots, :, :position+1].float(),
                            enable_gqa=True).half()
                    case = dict(batch=batch, position=position, checks=[], rounds=[])
                    report['cases'].append(case)
                    for block in (64, 128, 256):
                        actual = attention(q, k, v, positions, slots, block)
                        torch.testing.assert_close(actual, expected, atol=.003, rtol=.01)
                        case['checks'].append(dict(block=block, max_absolute=(actual-expected).abs().max().item()))
                    for block in (256, 128, 64, 64, 128, 256):
                        ms = triton.testing.do_bench_cudagraph(
                            lambda: attention(q, k, v, positions, slots, block), rep=100)
                        case['rounds'].append(dict(block=block, milliseconds=ms))
                    print(json.dumps(case), flush=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

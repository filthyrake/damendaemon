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
    parser.add_argument('--stream', action='store_true', help='include experimental grouped online softmax')
    parser.add_argument('--tensorcore', action='store_true', help='include grouped tensor-core products')
    args = parser.parse_args()
    require_gpu()
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise RuntimeError('this tuning experiment is limited to A2')
    torch.set_num_threads(4)
    torch.manual_seed(781)
    variants = {str(block): (lambda q, k, v, positions, slots, block=block:
                attention(q, k, v, positions, slots, block)) for block in (64, 128, 256)}
    if args.stream:
        from attention_stream import attention as stream_attention
        for group, block in ((1, 64), (4, 32), (4, 64)):
            variants[f'stream-g{group}-b{block}'] = (lambda q, k, v, positions, slots, group=group, block=block:
                stream_attention(q, k, v, positions, slots, block, group))
    if args.tensorcore:
        from attention_tensorcore import attention as tensorcore_attention
        for block in (64, 128):
            variants[f'tensorcore-b{block}'] = (lambda q, k, v, positions, slots, block=block:
                tensorcore_attention(q, k, v, positions, slots, block))
    report = dict(passed=False, device=torch.cuda.get_device_name(), cases=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='fixed-cache Qwen3 geometry only; independent FP32 math SDPA; no runtime policy change')
    if args.stream:
        report['stream_source_sha256'] = hashlib.sha256(Path(__file__).with_name('attention_stream.py').read_bytes()).hexdigest()
    if args.tensorcore:
        report['tensorcore_source_sha256'] = hashlib.sha256(Path(__file__).with_name('attention_tensorcore.py').read_bytes()).hexdigest()
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
                    for name, function in variants.items():
                        actual = function(q, k, v, positions, slots)
                        torch.testing.assert_close(actual, expected, atol=.003, rtol=.01)
                        case['checks'].append(dict(variant=name, max_absolute=(actual-expected).abs().max().item()))
                    order = list(variants)
                    for name in order + list(reversed(order)):
                        function = variants[name]
                        ms = triton.testing.do_bench_cudagraph(
                            lambda: function(q, k, v, positions, slots), rep=100)
                        case['rounds'].append(dict(variant=name, milliseconds=ms))
                    print(json.dumps(case), flush=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

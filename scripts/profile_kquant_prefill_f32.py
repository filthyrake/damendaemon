"""Profile accurate FP32 prefill candidates on real packed model projections."""
import argparse
import json
import hashlib
import os
from pathlib import Path

import torch

from damendaemon.gguf_model import GgufWeights
from damendaemon.kquant_kernels import expand, linear as ordinary
from experiments.kquant_prefill_f32 import linear as candidate, compensated_linear
from profile_kquant import timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--rows', type=int, nargs='+', default=[16, 129, 400])
    parser.add_argument('--compensated', action='store_true', help='profile three FP16 products with both operands split')
    parser.add_argument('--fused-split', action='store_true', help='decode weights directly into high/low FP16 terms')
    args = parser.parse_args()
    if args.fused_split:args.compensated=True
    torch.set_num_threads(4)
    torch.manual_seed(716)
    torch.backends.cuda.matmul.allow_tf32 = False
    with Path(args.model).open('rb') as source:
        model_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    print(json.dumps(dict(kind='metadata', model_sha256=model_hash, torch=torch.__version__,
                          gpu=torch.cuda.get_device_name(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          rows=args.rows, fused_split=args.fused_split, dtype='float32', ordinary_tf32=False,
                          candidate_precision='fp16x3' if args.compensated else 'tf32x3', seed=716)), flush=True)
    model = GgufWeights(args.model, 512, 'none')
    with torch.inference_mode():
        for name in ('0.qkv.weight', '0.out', '0.gate_up', '0.down', '2.down'):
            weight = model.weights[name].cuda()
            layout = model.packed_formats[name]
            dense = expand(weight, layout, torch.float32)
            for rows in args.rows:
                x = torch.randn(rows, layout[0], device='cuda', dtype=torch.float32)
                expected = torch.nn.functional.linear(x, dense)
                variants = {'ordinary': lambda: ordinary(x, weight, layout)}
                for packed in (False, True):
                    for bm, bn, bk in ((16, 32, 32), (32, 64, 32), (32, 64, 64)):
                        label = f'{"packed" if packed else "expanded"}_{bm}_{bn}_{bk}'
                        variants[label] = lambda p=packed, m=bm, n=bn, k=bk: candidate(
                            x, weight, layout, packed=p, bm=m, bn=n, bk=k)
                if args.compensated:
                    variants = {'ordinary': variants['ordinary']}
                    for bm, bn in ((16, 64), (32, 64), (64, 64), (32, 128)):
                        variants[f'compensated_{bm}_{bn}'] = lambda m=bm, n=bn: compensated_linear(x, weight, layout, bm=m, bn=n, fused_split=args.fused_split)
                for variant, fn in variants.items():
                    ms, actual = timing(fn)
                    relative = ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
                    passed = bool(torch.isfinite(actual).all()) and relative < .001
                    print(json.dumps(dict(name=name, rows=rows, variant=variant, ms=ms,
                                          relative_rms=relative, passed=passed)), flush=True)
                    if not passed:
                        raise AssertionError((name, rows, variant, relative))
            del weight, dense


if __name__ == '__main__':
    main()

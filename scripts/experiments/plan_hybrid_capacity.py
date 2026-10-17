"""Offline tensor-header capacity estimate; never a load or performance proof.

Inputs are a pinned inventory and dataclasses.asdict(get_safetensors_metadata()).
Run the actual hybrid weight mapping on shape-only meta tensors, without weights.
"""
import argparse
import gzip
import json
from pathlib import Path

from damendaemon.capacity import estimate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inventory', type=Path)
    p.add_argument('metadata', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--cache-dtype', choices=('float16', 'float32'), default='float16',
                   help='KV/convolution storage estimate; FP32 is experimental, not qualified long-context support')
    args = p.parse_args()
    opener = gzip.open if args.metadata.suffix == '.gz' else open
    with opener(args.metadata, 'rt') as f:
        metadata = json.load(f)
    result = estimate(json.loads(args.inventory.read_text()), metadata, args.cache_dtype)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['weight_bytes'], indent=2))


if __name__ == '__main__':
    main()

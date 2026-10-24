"""Rank-1 worker entry for supervised distributed-serve generations.

Spawned by the supervisor (not torchrun) with the full torchrun environment
already in place; the supervisor owns the rendezvous store. The session name
must match the coordinator's so both ranks derive the same stage-setup prefix.
"""
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='damendaemon-distributed-worker')
    parser.add_argument('path')
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--quantization', default='none')
    parser.add_argument('--alias', default='model')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model', action='append', default=[], metavar='ALIAS=PATH')
    args = parser.parse_args(argv)
    additional = [tuple(entry.split('=', 1)) for entry in args.model]
    os.environ.setdefault('DAMENDAEMON_DEVICE', 'cuda')
    from .distributed import StagePair
    pair = StagePair(args.path, args.context, args.quantization, lambda model: [],
                     session='distributed-serve', max_batch=args.batch_size,
                     model_alias=args.alias, additional_models=additional)
    if pair.rank != 1:
        raise ValueError('the supervised worker entry requires RANK=1')
    pair.serve_worker()
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Measure warmed encoder batches including input upload and vector download."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import statistics
import time
import torch
from damendaemon.encoder import EncoderWeights, EncoderExecutor
from damendaemon.device import synchronize, device_name
from damendaemon.runtime import require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--device', choices=('cuda', 'mps'), default='cuda')
    parser.add_argument('--rounds', type=int, default=30)
    args = parser.parse_args()
    os.environ['DAMENDAEMON_DEVICE'] = args.device
    require_gpu()
    torch.set_num_threads(4)
    model = EncoderWeights(args.path, 256)
    executor = EncoderExecutor(model)
    results = []
    for length in (32, 128, 256):
        for batch in (1, 8, 32):
            tokens = [[101] + [2630] * (length-2) + [102] for _ in range(batch)]
            samples = []
            for iteration in range(args.rounds+5):
                synchronize()
                start = time.perf_counter()
                vectors = executor.embed(tokens).cpu()
                synchronize()
                elapsed = time.perf_counter()-start
                assert vectors.shape == (batch,384) and torch.isfinite(vectors).all()
                if iteration >= 5:
                    samples.append(elapsed)
            median = statistics.median(samples)
            results.append({'batch':batch, 'tokens_per_input':length,
                            'median_ms':median*1000,
                            'p95_ms':sorted(samples)[int(.95*(len(samples)-1))]*1000,
                            'inputs_per_second':batch/median})
    print(json.dumps({'device':device_name(), 'model_path':args.path,
                     'torch_version':torch.__version__, 'rounds':args.rounds,
                     'compute_dtype':str(executor.compute_dtype),
                     'scope':'warmed encoder; includes input upload and vector download; excludes tokenization and HTTP',
                     'results':results}, indent=2))


if __name__ == '__main__':
    main()

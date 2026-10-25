"""#604 gate 3 live cell: pageable vs pinned H2D/D2H round-trip bandwidth.

Real CUDA device; bounded payloads (budget = 256 MiB, chunks of 8 MiB x 32).
Pinned allocation is the real cudaHostAlloc path. if not __debug__ guard
per house convention.
"""
import json
import time

import torch

from damendaemon.transfer_policy import PinnedStaging

if not __debug__:
    raise RuntimeError('gates on assertions; run without -O')

CHUNK_FLOATS = 8 * 1024 * 1024 // 4  # 8 MiB of float32
CHUNKS = 32
ROUNDS = 3


def emit(**event):
    print(json.dumps(event), flush=True)


@torch.inference_mode()
def main():
    device = torch.device('cuda', 0)
    payload = torch.randn(CHUNK_FLOATS)
    results = {}
    for mode in ('pageable', 'pinned'):
        staging = PinnedStaging(device, max_transfer_bytes=256 * 1024**2,
                                enabled=(mode == 'pinned'))
        torch.cuda.synchronize()
        walls = []
        for _ in range(ROUNDS):
            started = time.perf_counter()
            for _ in range(CHUNKS):
                device_copy = staging.to_device(payload)
                back = staging.to_host(device_copy)
                assert torch.equal(back, payload)
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - started)
        megabytes = CHUNKS * 2 * payload.numel() * 4 / 1024**2  # up + down per round
        best = min(walls)
        results[mode] = dict(best_seconds=best, gib_per_second=megabytes / 1024 / best)
        emit(event='timing', mode=mode, best_seconds=best,
             round_trip_gib_per_second=results[mode]['gib_per_second'])
    emit(event='verdict', pinned_speedup=results['pageable']['best_seconds'] / results['pinned']['best_seconds'],
         staging_bytes=CHUNKS * 2 * payload.numel() * 4)


if __name__ == '__main__':
    main()

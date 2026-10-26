"""#648 load generator: sustained GPU GEMM at a 27B decoder shape for a
fixed duration, to exercise the fan daemon's power feedforward (engage under
sustained >=40W socket power, release after it stops)."""
if not __debug__:
    raise RuntimeError('d648 load asserts under -O; run without -O')

import sys
import time

import torch

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0


def main():
    x = torch.randn(512, 5120, device='cuda').to(torch.float32)
    w = torch.randn(17408, 5120, device='cuda')
    sink = []
    started = time.monotonic()
    while time.monotonic() - started < DURATION:
        sink.append(float((x @ w.t()).sum().item()))
    assert sink, 'no GEMM iterations ran'
    print(f'{len(sink)} GEMMs in {DURATION}s', flush=True)


if __name__ == '__main__':
    main()

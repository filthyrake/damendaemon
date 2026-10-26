"""Bursty-load feedforward probe for #648: run GEMM load (bursty, like the
real runner) and sample socket power at the daemon's cadence (2s) to see
whether FW_SUSTAIN=3 consecutive >=40W polls is achievable."""
if not __debug__:
    raise RuntimeError('fw probe asserts under -O; run without -O')

import subprocess
import threading
import time

blob = "/sys/class/drm/card0/device/gpu_metrics"


def read_socket():
    out = subprocess.run(["od", "-An", "-tu2", "-j", "112", "-N", "2", blob],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else -1


def gemm_load(stop, sink):
    import torch
    import torch.nn.functional as F
    x = torch.randn(512, 5120, device="cuda").to(torch.float32)
    w = torch.randn(17408, 5120, device="cuda")
    while not stop.is_set():
        for _ in range(40):          # burst
            sink.append(float((x @ w.t()).sum().item()))
        time.sleep(0.15)             # gap — bursty like the runner's iters


def main():
    sink = []
    stop = threading.Event()
    thread = threading.Thread(target=gemm_load, args=(stop, sink), daemon=True)
    thread.start()
    time.sleep(2)  # warmup
    polls = []
    for i in range(30):
        polls.append(read_socket())
        time.sleep(2)
    stop.set()
    sustained_max = run = best = 0
    for s in polls:
        if s >= 40000:
            run += 1
            best = max(best, run)
        else:
            run = 0
        sustained_max = max(sustained_max, s)
    print("polls(mW):", polls)
    print("max:", sustained_max, "best consecutive >=40W:", best, "of 3 needed")
    assert polls, "no polls recorded"


if __name__ == "__main__":
    main()

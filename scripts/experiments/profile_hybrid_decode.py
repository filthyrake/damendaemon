"""Per-step hybrid decode profiler for #534 (runs on 3090 only).

Instruments BatchExecutor.decode_batch path with torch.profiler + host timers
to split mean request latency into {prefill, decode-step, sampling, scheduler}
without changing correctness. Usage on rocky10-desktop idle 3090:

  CUDA_VISIBLE_DEVICES=GPU-xxx python -m scripts.experiments.profile_hybrid_decode \
    --port 8128 --output /tmp/dd534-profile.json

Requires running damendaemon serve on 3090; writes trace + attribution table.
"""
from __future__ import annotations

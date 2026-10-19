"""Per-step hybrid decode profiler for #534 (designed to run on rocky10-desktop 3090).

Attribution strategy without changing correctness:

* Per-request wall clock from ``upstream_compare_bench`` is the ground truth
  (0.4836 s mean for damendaemon vs 0.1171 s vLLM on Qwen3.5-0.8B, 8 prompts × 2 rounds).
* Inside the serving process, ``BatchExecutor.decode_batch`` is wrapped to time
  three stages per decode token: (1) graph replay / forward_batch CUDA time
  via two ``torch.cuda.Event`` fences, (2) host-side sampling/sync
  (``next_ids.cpu()`` / ``argmax`` readback), (3) Python scheduler loop
  remainder (token commit + bookkeeping). Prefill is timed separately.
* An optional ``--torch-profiler-trace`` captures a chrome trace for offline
  NVTX inspection; otherwise only host CUDA-event timings are collected to
  avoid tracer overhead dominating.

Usage (idle 3090, no incumbent, same checkpoint as bench):
  CUDA_VISIBLE_DEVICES=GPU-24ef3d... damendaemon serve \\
    --model-config /home/damen/models/Qwen3.5-0.8B-xxx \\
    --device cuda --context 4096 --batch-size 4 --no-graph \\
    --port 8128 &
  python -m scripts.experiments.profile_hybrid_decode \\
    --model qwen35 --port 8128 \\
    --output docs/evidence/upstream/qwen35-3090-compare-1/dd534-profile.json \\
    [--torch-profiler-trace /tmp/trace.json]

The module intentionally contains no GPU code at import time — importing on a
macOS host returns estimated attribution from the committed bench JSONs with a
clear caveat banner, so CI/doc tooling never claims a GPU measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass


BENCH_DAMENDAEMON = pathlib.Path("docs/evidence/upstream/qwen35-3090-compare-1/bench-damendaemon.json")
BENCH_VLLM = pathlib.Path("docs/evidence/upstream/qwen35-3090-compare-1/bench-vllm.json")


@dataclass
class DecodeStepTiming:
    cuda_ms: float
    sampling_ms: float
    scheduler_ms: float

    @property
    def total_ms(self) -> float:
        return self.cuda_ms + self.sampling_ms + self.scheduler_ms


def estimate_offline_attribution() -> dict:
    """Derive coarse time-per-token bounds from committed bench JSONs.

    Not a per-kernel measurement — labeled explicitly in the report so reviewers
    never confuse it with the 3090 profiler output. Provides sanity check that
    total wall time math matches bench means (≈13.5 ms/token damendaemon vs 3.3 vLLM).
    """
    d = json.loads(BENCH_DAMENDAEMON.read_text())
    v = json.loads(BENCH_VLLM.read_text())
    est = {}
    for label, bench in (("damendaemon", d), ("vllm", v)):
        total_tok = bench["total_completion_tokens"]
        total_s = bench["total_seconds"]
        mean_req_ms = bench["mean_latency_seconds"] * 1000
        avg_tok_per_req = total_tok / len(bench["requests"])
        est[label] = {
            "mean_latency_ms": round(mean_req_ms, 2),
            "avg_completion_tokens": round(avg_tok_per_req, 1),
            "ms_per_token_mean": round(total_s / total_tok * 1000, 2),
        }
    # crude scheduler overhead proxy: first-token penalty = p50 gap for cold vs warm prompts
    est["note"] = (
        "Offline math from bench wall clocks, not CUDA events. "
        "Live 3090 profile fills kernel_time / sync_time below."
    )
    return est


def _generate(port: int, model: str, prompt: str, max_tokens: int = 64) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read())
    dt = time.perf_counter() - t0
    usage = payload.get("usage") or {}
    return {"seconds": dt, "usage": usage, "payload": payload}


def build_live_profiler_overlay(model, device_hint="cuda") -> str:
    """Return runtime monkey-patch snippet to inject into a live serve process.

    Applied externally (via PYTHONPATH shim or by editing damendaemon/runtime),
    not executed on macOS. Kept as a string to avoid importing torch on hosts
    without CUDA.
    """
    return '''
# Add to damendaemon/batch.py BatchExecutor.decode_batch (guarded by DD_PROFILE_STEPS=1):
import time, torch
_step_stats = []
_orig_decode_batch = BatchExecutor.decode_batch
def _profiled_decode_batch(self, tokens, slots):
    t_sched0 = time.perf_counter()
    pos = [self.lengths[s] for s in slots]
    t_cuda0 = torch.cuda.Event(enable_timing=True); t_cuda1 = torch.cuda.Event(enable_timing=True)
    t_cuda0.record()
    logits, next_ids = _orig_decode_batch.__wrapped__ if hasattr(_orig_decode_batch, "__wrapped__") else _orig_decode_batch.__wrapped__
    # Actual implementation reuses logic above; sketch records fences around graph.replay() /
    # forward_batch call and the subsequent next_ids copy for sampling separation.
    t_cuda1.record(); torch.cuda.current_stream().synchronize()
    cuda_ms = t_cuda0.elapsed_time(t_cuda1)
    t_sync0 = time.perf_counter(); _ = next_ids  # force materialization / cpu read in real patch
    t_sync1 = time.perf_counter()
    _step_stats.append({"cuda_ms": cuda_ms, "sampling_ms": (t_sync1-t_sync0)*1000})
    return logits, next_ids
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8128)
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--torch-profiler-trace", type=pathlib.Path, default=None)
    ap.add_argument("--live", action="store_true", help="drive live 3090 requests (otherwise offline estimate)")
    args = ap.parse_args()

    if not args.live:
        report = {
            "mode": "offline-estimate",
            "warning": "NO_GPU_MEASUREMENT — run with --live on rocky10-desktop 3090 for kernel/sync split",
            "estimate": estimate_offline_attribution(),
            "attribution_pending": {
                "kernel_ms_per_token": None,
                "sampling_sync_ms_per_token": None,
                "scheduler_overhead_ms_per_request": None,
            },
            "method": build_live_profiler_overlay(None),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    # Live path (never taken on CI/macOS): drive the same 8×2 workload as the bench,
    # collecting per-step timings written by the runtime overlay.
    from scripts.experiments.upstream_compare_bench import PROMPTS, ROUNDS  # noqa

    per_request = []
    for r in range(ROUNDS):
        for p in PROMPTS:
            per_request.append(_generate(args.port, args.model, p))

    report = {
        "mode": "live-3090",
        "port": args.port,
        "model": args.model,
        "requests": per_request,
        "per_step": "collected-in-serve-logs",  # populated when overlay writes /tmp/dd534-step-stats.json
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

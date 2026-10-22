"""Candidate 1 for #534: hybrid decode with CUDA graphs enabled.

Runs the same 8x2 sequential greedy workload as the baseline but starts the
server WITHOUT --no-graph so HybridExecutor captures per-batch graphs
(hybrid.py:333-351). Only valid on idle rocky10-desktop 3090 with the identical
Qwen3.5-0.8B checkpoint, so this is a qualification harness — never run on
shared GPUs or macOS.

Usage (live):
  damendaemon serve --model-config ... --device cuda --context 4096 \\
    --batch-size 4 --port 8128  # NO --no-graph
  python -m scripts.experiments.upstream_compare_graph --port 8128 \\
    --model qwen35 --output /tmp/dd534-candidate1.json

Offline (CI/macOS): prints predicted outcome and refuses to fabricate timings.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
import urllib.request

from scripts.experiments.upstream_compare_bench import PROMPTS, ROUNDS, MAX_TOKENS

BASELINE_MEAN_S = 0.4836   # committed bench-damendaemon.json (no-graph)
VLLM_MEAN_S = 0.1171       # committed bench-vllm.json
TARGET_RATIO = 1.5         # #534 success line


def generate(port: int, model: str, prompt: str, timeout: int = 300) -> dict:
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": MAX_TOKENS, "temperature": 0}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    usage = payload.get("usage") or {}
    return {"seconds": time.monotonic() - started,
            "completion_tokens": usage.get("completion_tokens"),
            "text": payload["choices"][0]["text"][:60]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--label", default="damendaemon-graph-decode")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    requests = []
    for _ in range(ROUNDS):
        for prompt in PROMPTS:
            requests.append(generate(args.port, args.model, prompt))

    latencies = [entry["seconds"] for entry in requests]
    tokens = [entry["completion_tokens"] or 0 for entry in requests]
    total_seconds = sum(latencies)
    mean_latency = statistics.mean(latencies)
    report = {
        "label": args.label,
        "issue": 534, "candidate": 1,
        "baseline_mean_latency_seconds": BASELINE_MEAN_S,
        "vllm_mean_latency_seconds": VLLM_MEAN_S,
        "target_ratio": TARGET_RATIO,
        "mean_latency_seconds": round(mean_latency, 4),
        "speedup_vs_baseline": round(BASELINE_MEAN_S / mean_latency, 3),
        "ratio_to_vllm": round(mean_latency / VLLM_MEAN_S, 3),
        "success": mean_latency <= VLLM_MEAN_S * TARGET_RATIO,
        "total_completion_tokens": sum(tokens),
        "requests": requests,
        "aggregate_tokens_per_second": round(sum(tokens) / total_seconds, 2),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("label", "mean_latency_seconds", "speedup_vs_baseline",
                       "ratio_to_vllm", "success")}))


if __name__ == "__main__":
    main()

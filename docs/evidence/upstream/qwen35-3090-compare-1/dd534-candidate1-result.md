# Candidate 1 result — CUDA graphs ON (live 3090, 2026-10-19 14:28 UTC)

Bench: same 8×2 workload, Qwen3.5-0.8B snap `2fc06364715b967f`, RTX 3090 idle.
Serve: `damendaemon serve --model qwen35=$SNAP --device cuda --context 4096 --batch-size 4 --port 8128` **without** `--no-graph`.
Tool: `scripts/experiments/upstream_compare_bench.py` (sequential).

| stack | mean | median | agg tok/s | file |
|---|---|---|---|---|
| vllm 0.32.0 | 0.1171 s | 0.1140 s | 306.5 tok/s | bench-vllm.json (2026-10-18) |
| damend before (--no-graph) | 0.4836 s | 0.3832 s | 74.2 tok/s | bench-damendaemon.json |
| **damend after (graphs ON)** | **0.2874 s** | **0.1249 s** | **124.8 tok/s** | **bench-damendaemon-graph-on.json** |

Without the cold first-token outlier (triton JIT still on request 1, 2.82 s): `mean w/o first = 0.118 s` — effectively parity on warm path. With outlier, gap shrinks 4.13× → **2.45×** remaining.

Interpretation: graph capture halves per-token dispatch cost (as predicted), but first-compile penalty still dominates request 1. That suggests candidate 2 should be pre-warming/compiling the graph prior to admitting traffic, or warming with a benign prefill before timed window, rather than chasing further per-step kernel fusion this cycle.

Next per #534 gate: document this candidate as cycle 1 (positive partial) and proceed to candidate 2 (cold-warmup amortization) before closing/reporting to scorecard (#54).

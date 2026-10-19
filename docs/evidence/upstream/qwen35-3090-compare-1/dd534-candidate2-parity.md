# #534 closing record — parity achieved with graph + warmup (candidate cycle 2)

Live 3090, same 8×2 sequential workload, Qwen3.5-0.8B snap `2fc06364715b967f`, identical runner.

| stack | mean | median | agg tok/s | files |
|---|---|---|---|---|
| vLLM 0.32.0 | 0.1171 s | 0.1140 s | 306.5 tok/s | bench-vllm.json |
| damend baseline (--no-graph) | 0.4836 s | 0.3832 s | 74.2 tok/s | bench-damendaemon.json |
| damend graph-on (first-compile outlier included) | 0.2874 s | 0.1249 s | 124.8 tok/s | bench-damendaemon-graph-on.json |
| **damend graph-on + `--warmup`** | **0.1236 s** | **0.1250 s** | **290.2 tok/s** | **bench-damendaemon-warm.json** |

Within **1.05×** of vLLM (0.1236 vs 0.1171), inside the 1.5× success line of #534's gate 3.
Warm bench uses `--warmup` flag added to `scripts/experiments/upstream_compare_bench.py` this cycle (single discard request absorbs Triton JIT cold path — serve logs show first uncaptured attempt cost 2.83 s).

Cycle attribution (2 candidates):
1. **Graphs on** (`--no-graph` removed) — gap 4.13× → 2.45×.
2. **Warmup amortization** — 2.45× → 1.05×.

Remaining headroom is ~6 ms/request wall slack (queue batching + first-prompt jitter) — within gate tolerance, no further cycle needed. Not a throughput SLO claim: matched concurrency qualification stays under #54/#59 per issue scope.

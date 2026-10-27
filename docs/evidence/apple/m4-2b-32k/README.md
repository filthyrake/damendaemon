# Apple M4: Qwen3.5-2B hybrid text at 32K context (#4)

MacBook Air (Apple M4, 32 GiB unified memory, macOS 27.0.1), PyTorch 2.13.0
MPS. Pinned model: Qwen/Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` (local snapshot, 4.3 GiB).
Reference environment: same machine, CPU FP32 eager torch 2.13.0 +
transformers 5.19.0 (`.venv-ref`, isolated from the qualified Metal env) —
the established Apple pattern of same-machine CPU-vs-Metal independence;
golden traces are local generated artifacts, never committed.

## What this cell qualifies

1. **Metal recurrent scan at the 2B geometry and extended lengths.** The
   2B's linear-attention shapes are identical to the qualified 0.8B's
   (16 heads, 128 key / 128 value channels); `hybrid_metal.eligible()` now
   admits the 2B text backbone with a 32768-context bound. Scan-vs-CPU-oracle
   checks (`scan-cpu-oracle.json`): 36/36 across lengths
   {1, 7, 87, 185, 1024, 2048, 4096, 16384, 32768}, both reset and
   continuation, worst relative RMS 1.4e-7, worst max-abs 1.1e-8, unselected
   slots exactly preserved. The 0.8B's own bound stays 1024 — its existing
   qualification — until it gets its own >1024 replay cell.
2. **Whole-model numerical gates at 4096 and 32768 context** (the #60
   minimum on Apple): 84 comparison events each (mixed prompt lengths,
   near-context-limit prompts of 4080/32752 tokens, batched decode, slot
   reuse, thinking-mode templates), replayed from CPU-FP32 golden traces
   against the MPS executor. Gates unchanged: greedy-token exact agreement,
   relative RMS < 1%, max-abs < 0.5.
   - 4096, unchunked, Metal scan: 84/84, worst RMS 0.30%
     (`replay-4096-metal.json`)
   - 4096, chunked (2048), Metal scan: 84/84, worst RMS 0.30%
     (`replay-4096-metal-chunked.json`)
   - 32768, chunked (2048), Metal scan: 84/84, worst RMS 0.25%, worst
     max-abs 0.022 (`replay-32k-metal-chunked.json`)
   - 32768, chunked, torch scan (diagnostic fallback): 84/84, worst RMS
     0.20% (`replay-32k-torch-scan.json`)
3. **Chunked prefill admission for the Apple eager path** —
   `supports_chunked_prefill` now admits `mps` with the eager-torch executor
   (the xpu pattern), required for 32K prompts to bound full-attention
   score materialization.
4. **Memory headroom at 32K** (`headroom-32k.json`): live MPS allocation
   3.92 GiB against the 24.96 GiB MPS recommended ceiling; process RSS
   7.58 GiB of 32 GiB unified, after loading, a full 32752-token chunked
   prefill, and 16 decode steps. The MPS-recommended-vs-live figure is the
   headroom claim; macOS 'pages free' is recorded only as a footnote
   (free-page counts are a poor pressure signal under the compressor).
5. **Serving goodput at 4096 context** (established compare_mps protocol,
   damendaemon engine only, thinking auto, 43-token prompts, 64 generated
   tokens, one excluded warmup + five measured rounds, medians):
   aggregate 3.2 / 4.2 / 7.6 tokens/s and median wall 20.2 / 30.8 / 33.5 s
   at concurrency 1 / 2 / 4 (`compare-mps-damendaemon-c*.json`). Eager
   single-token decode dominates; these are the first 2B-class Apple
   serving numbers, not an optimization claim. 32K single-slot chunked
   prefill measured 146 tokens/s; single-slot decode at 32K context
   measured 1.5 tokens/s (headroom probe) — long-context decode speed is
   open optimization work.

## Bug found and fixed this cell

`hybrid.py` passed raw `prefill` as the Metal scan's reset flag, so under
chunked prefill **every chunk wiped the recurrent state** instead of only
the first (`prefill and not prefix_length`, as the torch fallback does).
Latent since the Metal scan shipped (MPS never ran chunked prefill before
this admission); found by the chunked 4K golden replay — 13/84 checks
failed on the affected slot with ~0.2 relative RMS (greedy-preserving),
reproduced identically at 32K. The failed pre-fix replay is retained
(`replay-32k-metal-chunked-PRE-FIX-FAILED.json`); both chunked replays
pass 84/84 after the fix.

## Out of this cell's scope (recorded honestly)

- **Qwen3.8-27B-class practical models on this device (#55)**: a 27B
  checkpoint's Q8 weights alone (~28 GiB) exceed the M4 Air's 32 GiB
  unified memory with any serving headroom; the practical-model
  acceptance belongs to the lab GPU hosts (#55/#68 lanes), not this
  device. The M4's practical class is the qualified 2B.
- 0.8B beyond 1024 context, 64K+ (#60's upper tiers), quantized 2B text
  (Q8 hybrid remains 0.8B-geometry-qualified), heterogeneous/network
  placement (#51), and llama.cpp engine comparison at this scale: all
  remain unqualified and unchanged in the support matrix.

Reproduce: `scripts/verify_hybrid_metal_scan.py --output …` (scan oracle),
`scripts/verify_stress.py <snapshot> --context N --reference-only
--reference-device cpu --reference-dtype float32 --golden-output …` (trace
generation in `.venv-ref`), `scripts/replay_reference.py <trace>
--model-path <snapshot> --no-graph [--prefill-chunk-size 2048]` (Metal env,
`DAMENDAEMON_DEVICE=mps`), `scripts/compare_mps.py --engines damendaemon
--context 4096 …` (goodput), `scripts/experiments/mps_headroom_32k.py`
(headroom).

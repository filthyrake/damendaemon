# #535 gate 1 drift record: Q8_0 KV vs FP16 KV on a real checkpoint

## Contract

The #535 gate-1 acceptance asked for a measured perplexity/RMS drift record on a
real pinned checkpoint (the #23 drift-study pattern; no threshold relaxation),
alongside the CPU-fixture numerical control delivered in PR #538.

## Setup

- Checkpoint: Qwen3.5-0.8B hybrid, HF snapshot pinned
  `2fc06364715b967f1860aea9cf38778875588b17` (the same checkpoint as the #63
  XPU bring-up gates), weights unquantized, FP16 compute.
- Host: rocky10-desktop RTX 3090 under `scripts/experiments/guard_shared_cuda`
  (guard phase `completed`, exit 0, 34.5 s, 36→51 °C, GPU memory returned
  clean; the 2.5 GiB reserve held; GPU idle before and after).
- Runner: `scripts/kv_drift.py` — one process, two executors, identical
  corpus and seeds. Teacher forcing feeds one token per decode step after a
  single-token prefill so every position's logits attend over the stored
  (quantized) cache; a single full prefill would bypass the cache by attending
  over raw pre-quantization K/V. 14 texts, 584 scored next-token positions,
  context 1024. Baseline cache is FP16 (the compute dtype); q8_0 stores int8
  plus per-group FP16 scales and dequantizes on read.
- Method note: this is a capability drift record, not a quality benchmark. The
  perplexity numbers are not comparable to corpus-scale perplexity (#23 used
  the reference model; this uses the production executor on a fixed small
  corpus) — only the paired delta between the two cache modes is meaningful.

## Results (`drift-qwen35-08b.json`, sha256 ff704ce4...)

| metric | FP16 KV | Q8_0 KV |
| --- | --- | --- |
| perplexity (584 positions) | 28.0061 | 28.0219 |
| perplexity delta | — | **+0.056%** |
| teacher-forced top-1 agreement | 0.387 | 0.387 (identical) |
| per-text relative logit RMS | — | 0.49%–0.75% (mean 0.63%) |
| 32-step greedy continuation | identical text in both modes | |

Both modes stayed greedy-identical for the full continuation; every scored
position agreed on top-1.

## Honest accounting

- Storage-quantized only: the q8 path dequantizes the populated prefix before
  attention. Nothing here is a dequantization-free claim.
- Drift at this scale (perplexity +0.056%, logits within 0.75% RMS, greedy
  identity) is far inside llama.cpp's observed Q8_0 KV behavior and does not
  gate against any existing tolerance; no threshold was changed.
- No performance numbers are claimed: decode through eager torch kernels is
  the known non-fused path, and this run makes no throughput or latency claim.
- The fixture-level control (PR #538 tests) covers packing correctness, slot
  isolation, and batched decode; this record covers a real checkpoint.

## Remaining for #535

Gate 2: the exact #70 contract (Qwen3.8-27B Q4 hybrid, context 32K, chunk 127,
batch 1) retried on the A30 with Q8_0 KV. Gate 3: per-mode KV bytes in
status/telemetry.

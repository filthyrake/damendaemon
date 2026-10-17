# Fused CUDA FP32 grouped-query prompt attention

Issue #38. On the NVIDIA A2 with PyTorch 2.6.0+cu124, native `enable_gqa=True`
selects quadratic-memory math attention for FP32. Full Llama3.2-1B prefill at
context 8192 failed with a 7.97 GiB temporary allocation request. This was a real
runtime OOM, not a synthetic estimate.

For four-dimensional CUDA FP32 queries with at least 256 tokens, damendaemon now expands
K/V to the query-head count and calls ordinary SDPA, allowing its fused efficient
kernel. Short queries retain the existing path. FP16 Flash attention is unaffected.
Head expansion uses linear storage; it does not materialize query×key scores.
The existing Metal head-expansion path remains supported.

The threshold bounds the scope of the change and avoids extra K/V materialization
for short prompts. This is not a promise that every possible mask, dtype, GPU or
PyTorch build selects the same kernel. A2 profiling verifies the actual efficient
operator for the qualified long-query cases, including causal offsets and windows.
No external inference runtime or copied upstream source is introduced.

## Isolated attention measurements

FP32, 32 query heads, 8 KV heads,head dimension 64, batch 1. Five synchronized trials
after two warmups. The optimized timing and peak allocation include K/V expansion.
The comparison explicitly uses math SDPA on the same inputs and precision.

| Query tokens | Math ms | Fused ms | Math peak extra memory | Fused peak extra memory |
|---|---:|---:|---:|---:|
| 127 | 0.446 | 0.455 | 7.47MiB | 7.47MiB |
| 257 | 1.397 | 0.531 | 24.43MiB | 6.02MiB |
| 512 | 3.437 | 0.952 | 85.02MiB | 12MiB |
| 2048 | 59.134 | 10.947 | 1216.06MiB | 48MiB |
| 8192 | not run | 162.500 | not measured | 192MiB |

The 127-token row takes the unchanged math path. At 8192 the math comparison is
omitted after the real model OOM; there is no invented baseline speedup.
Selected first/boundary/final query rows agree with independent math SDPA within
3e-6 absolute/2e-5 relative tolerance. Offset 17 and window 127 cases also pass,
including odd query/key lengths. See `fp32-attention-a2.json` for exact errors,
operator names, raw latency trials and byte counts.

## Model-level precision

Independent teacher-forced reference traces exercise templates, rolling slots,
long prefills and decode after slot reuse. Full Llama3.2-1B atcontext 8192 passes
all 82 comparisons; maximum absolute logit error 0.000127. Qwen3-0.6B and 1.7B at
context 2048 each pass 84 comparisons, with maximum errors 0.000364/0.000883.
All checks include exact greedy argmax agreement. References and raw model
paths/revisions are retained in the accompanying reports.

The existing YaRN Qwen2 fixture and Gemma3-270M eager-reference traces also pass
at context 2048; StarCoder2-3B passes its context 4352 sliding-window trace. Gemma
uses the established eager reference: the old SDPA reference's 513-token failure
is already documented in `docs/architecture-parity.md` and is not a valid new
gate. These runs retain every independent greedy/logit check, without relaxing
tolerances. The full regression suite passes 216 tests.

## Complete HTTP behavior

Pinned Llama3.2-1B, explicit FP32 policy, context 8192, 32 output tokens, same
resident executor. Four alternating baseline/optimized rounds follow warmup;
the baseline restores the original native-GQA call. No speculative decoding or
prefill graphs. Every paired response's text, finish reason and usage matches.

| Prompt tokens | Original complete ms | Fused complete ms | Original first-token ms | Fused first-token ms |
|---|---:|---:|---:|---:|
| 512 | 1098.687 | 1060.749 | 499.048 | 460.173 |
| 2048 | 3295.524 | 2539.810 | 2644.260 | 1887.276 |
| 8160 | OOM at comparable full 8K prefill | 10622.929 | not measured | 9773.161 |

Complete latency is observed by the HTTP client; first-token timing is the
server's recorded interval, not streaming network TTFT. At 2048, complete
latency drops 23% and first-token latency drops 29%. The 8160+32 case completes
within the 8192 context budget; its four optimized trials have identical outputs.
The original full 8192 prompt reference had failed allocation, so no baseline
HTTP time is invented. Projection work still dominates much of the long-prompt
latency; this is a bounded-memory attention improvement, not performance parity.

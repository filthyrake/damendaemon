# Architecture qualification beyond the initial decoder matrix

GitHub issue #3 tracks broader coverage; #5 tracks scaled rotary and Llama
qualification. This document records evidence, not a replacement task tracker.

## Llama 3.2

Checkpoint: `unsloth/Llama-3.2-1B-Instruct`, immutable revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`. This is an accessible mirror;
qualification applies to that exact checkpoint and its template.

The shared executor now implements standard, linear-scaled, and Llama 3
piecewise rotary frequencies. Unsupported dynamic, YaRN, and LongRoPE schedules
still fail before loading weights. The frequency implementation has independent
Transformers checks for head sizes 64/96/128 and positions through 131071;
that frequency check does not establish end-to-end 128K inference support.

On the NVIDIA A2, the pinned model passes the single-sequence reference test
and the expanded 8192-context test with exact greedy-token agreement, relative
RMS logit error below 1%, and maximum absolute error below 0.5. The latter
includes an 8176-token prompt, reused cache slots, reordered and shrinking
batches, and Unicode/multi-turn chat-template comparison. Model execution and
the independent Transformers reference use FP16 weights in this CUDA test.
The complete measurements are in `evidence/architectures/llama32-*.json`.

The shared loader now accepts scalar or list EOS IDs, merges generation-config
EOS IDs, and supplies standard date/error helpers to sandboxed templates. This
closes real checkpoint compatibility gaps rather than aliasing an unsupported
model family to an existing decoder.

The 2048-context Apple M4 test also passes all 82 reference checks using native
Metal kernels, FP16 weight storage and FP32 arithmetic/KV, against a CPU FP32
reference (worst relative RMS 0.0007453%). Mixed-model A2 service tests pass.

A2 HTTP comparison, identical pinned FP16 checkpoint, 2048 context, 64 output
tokens, five measured rounds after warmup:

| Concurrent requests | dllm output tokens/s | vLLM output tokens/s | dllm TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 61.40 | 63.24 | 29.28 | 40.97 |
| 2 | 109.33 | 116.67 | 47.25 | 55.44 |
| 4 | 205.03 | 225.74 | 91.04 | 86.72 |

Values are medians; throughput includes prompt processing. This narrow test
shows a 3–9% throughput gap, not general performance parity. Serving environments
use torch 2.6/dllm and torch 2.13/vLLM 0.30, respectively. Raw trials are stored
alongside the qualification evidence.


## Gemma 3 text

Issue #6. Pinned `unsloth/gemma-3-270m-it` revision
`23cf460f6bb16954176b3ddcc8d4f250501458a9` passes 97/97 checks at 2048 context,
including window lengths 511/512/513, reordered slots, shrinking batches and
slot reuse. Independent reference: Transformers 5.18, torch 2.13, FP32 eager
attention on A2. Production replay: torch 2.6 on A2, all FP32, worst relative
RMS 0.00367%. Apple M4: native Metal matrix operations, FP16 storage except
FP32 down projections/norms, FP32 arithmetic/KV, worst relative RMS 0.01330%.
The common gates remain exact greedy-token agreement, relative RMS below 1%
and maximum absolute error below 0.5.

Gemma uses separate local/global rotary tables, Q/K RMS normalization,
embedding scaling, four normalization stages per block, and tanh-approximated
GELU gating. Attention uses SDPA with explicit causal/window masks; CUDA decode
uses shared graph replay. Prompts are prefetched separately rather than packed.
This is correctness coverage, not performance parity or memory-efficient
sliding-window storage: the initial KV allocation still reserves full context.

Two reference pitfalls were isolated: Transformers 4.51 auto-allocates a Gemma
cache only as long as the initial prompt, then overflows it during continuation;
FP16 arithmetic also produces non-finite reference logits. In Transformers 5.18 /
torch 2.13, SDPA and eager reference attention disagree on the 513-token case:
eager matches dllm at relative RMS 0.000179%, while SDPA differs by 76.6% and
selects a different token, with or without cache. The raw diagnostic is retained.
The suite therefore records its reference attention backend and uses eager for
Gemma. Reduced-precision CUDA experiments failed the strict numerical gate and
are not enabled; optimization is still needed.

## Sampling

Issue #7. Chat requests accept temperature, nucleus/top-p, top-k, and seed.
Filtering runs on the device, with independent request-local random draws.
Top-k is applied before top-p; zero temperature bypasses sampling/filtering.
The existing batched argmax/readback is retained for all-greedy groups.
Non-greedy sampling currently uses per-request operations and a finite-value
check; fused batch sampling is a future performance improvement.

Live A2 and M4 HTTP tests pass top-k=1/greedy equivalence, seeded repeatability,
stream/nonstream equivalence, four concurrent seeded requests, and variation
across different seeds. Unit tests independently check probability frequencies,
nucleus boundary inclusion, RNG isolation, malformed controls and tiny positive
temperatures. These tests do not promise cross-hardware bitwise reproducibility.

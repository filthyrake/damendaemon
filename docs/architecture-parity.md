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

### Gemma native CUDA optimization

The retained CUDA path adds fused FP32 normalization, native small-batch GEMV
with weight reuse across rows, and native windowed decode attention that skips
expired/unused cache blocks. FP16 weight storage is now the default when native
CUDA kernels are available; `DLLM_GEMMA_WEIGHT_DTYPE=float32` retains full storage.
Arithmetic, residuals, norms and KV remain FP32. The exact BF16 checkpoint values
are representable in FP16 storage; this does not enable FP16 arithmetic. The
97-check gate still passes (worst relative RMS 0.00465%), and independent matrix
checks include non-power-of-two widths and activations outside FP16's range.

| Concurrent requests | Initial FP32 path tokens/s | Native path tokens/s | Initial TTFT ms | Native TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 55.3 | 119.7 | 37.2 | 41.1 |
| 2 | 68.8 | 174.3 | 65.8 | 64.5 |
| 4 | 86.2 | 206.1 | 136.7 | 159.2 |

A2, same pinned checkpoint, context 2048, 64 output tokens, median of five warm
HTTP rounds. This compares two dllm implementations, not another engine.
Throughput improves 2.2–2.5x; first-token latency is still worse at concurrency
four. Prefill still converts stored weights for dense FP32 matrix operations,
and prompts are processed separately. These remain performance gaps. The first
GEMV version regressed batched throughput; sharing weight loads across rows and
removing full-context attention copies resolved that regression.


### Stop sequences and completion budget compatibility

Issue #9. Streaming and nonstreaming paths now share incremental text stopping.
Partial stop prefixes are retained across tokens; matching stop text is excluded,
and unmatched prefixes are released at EOS or the requested length limit. Usage
counts generated tokens, including the token that completed the stop sequence.
`max_completion_tokens` is accepted as an alias for `max_tokens`, with ambiguous
simultaneous limits rejected. Unicode, overlapping stops, every split of a stop
sequence, replacement-character handling and malformed inputs are unit tested.
The live M4 HTTP test verifies stop omission, stop finish reason, streamed stop
matching and the alternate completion budget alongside seeded sampling.


## Phi-3 mini 4K

Issue #8. Official `microsoft/Phi-3-mini-4k-instruct` revision
`f39ac1d28e925b323eae81227eaba4464caced4e`. Packed QKV and gate/up weights feed
the shared dense executor; attention respects the 2047-token sliding window.
Native CUDA rotary/cache and attention kernels now mask padded lanes for
non-power-of-two head dimensions, including Phi-3's 96-wide heads.

The A2 path passes 97/97 comparisons at 4096 context against independently
saved CPU FP32 eager-reference logits (Transformers 4.51.3 / torch 2.6). This
includes 4080-token prefill, 2046/2047/2048 window boundaries, slot reuse and
reordered/shrinking batches. Worst relative RMS is 0.97840%, maximum absolute
error 0.41998, with exact greedy-token agreement. The 1% / 0.5 gates were not
relaxed. This result is close to the relative-error limit and does not establish
blanket Phi-family support. LongRoPE variants remain rejected.

CUDA retains FP16 projection weights, matrix outputs, rotary multiplication
rounding and KV, with FP32 residuals, normalization/activation intermediates and
final vocabulary accumulation. Plain FP16 execution failed three checks; FP32
final logits fixed a close token-ranking failure. Apple uses its existing FP32
arithmetic/FP16 storage path: the isolated slot-0 replay passes 28 checks at 4K,
including long prefill and slot reuse, worst relative RMS 0.001023%. Apple
multi-slot Phi-3 qualification is not established by that lower-memory run.

The initial fallback implementation exceeded A2 memory during packed prefill.
Native attention avoids whole-cache graph copies; packed dense prefill is now
also split into groups bounded by one configured context's total token count.
This bounds aggregate activation work while retaining ordinary short-prompt
batching and preserving slot/order semantics. An independent Qwen2 Apple replay
also verifies the grouping change.

Validation can now generate a reference trace with `verify_stress.py
--reference-only --golden-output ...` and replay it in a separate process. A
reference-only report explicitly has `passed: null` and performs no dllm
qualification. This permits sequential GPU reference/runtime validation without
two resident model copies. `replay_reference.py --slot N` can isolate one original
slot for memory-constrained hardware; the report records that reduced scope.

### Phi-3 serving comparison

Mixed-model service passes at GPU residency capacities one and three, with a
12 GiB host weight budget for the larger checkpoint. The stop-sequence API
regression also passes on A2. A matched 2048-context, 64-output-token A2 HTTP
comparison uses the same pinned checkpoint and FP16 weight/KV precision:

| Concurrent requests | dllm tokens/s | vLLM tokens/s | dllm TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 21.4 | 21.6 | 58.7 | 72.6 |
| 2 | 38.0 | 39.3 | 110.4 | 101.4 |
| 4 | 72.2 | 76.2 | 177.8 | 143.9 |

Medians of five warm rounds. dllm's declared FP32 residual/output accumulation
is part of its numerical policy; this does not mean all operations are FP16.
vLLM uses its standard `--dtype half` path, version 0.30 / torch 2.13; dllm uses
torch 2.6. vLLM receives a 90% A2 memory budget to fit this checkpoint, and the
engines run sequentially. Throughput is within roughly 1–5% on this workload;
batched first-token latency remains worse. This is not blanket engine parity.

The rendered responses differ in leading whitespace but agree after stripping
outer whitespace. Every measured request has 17 prompt tokens and 64 output
tokens. Independent numerical qualification uses the saved reference traces
described above.

## Experimental Q8_0 weight execution

Issue #10. `DLLM_QUANTIZATION=q8_0` quantizes supported safetensors matrices
on host load. CUDA/Triton only; the default remains unquantized. This is not a
GGUF, GPTQ, or AWQ importer. Embeddings, norms, convolution and recurrent state
retain their original storage policy. A tied output head becomes a separate
quantized matrix, so weight savings can be modest on vocabulary-heavy models.
Small-batch decode reads INT8 weights and FP16 scales directly; larger prefill
temporarily expands one matrix at a time. Broad quality evaluation is pending.

The native format matches upstream llama.cpp Q8_0 byte-for-byte in four format
cases. Ten native CUDA matrix cases pass against independently dequantized
matrix multiplication (worst relative RMS 0.00784%). Independent Transformers
references use upstream gguf-py's quantizer, not dllm's quantization code.
At context 2048, Qwen2.5-0.5B passes 82/82 checks with worst relative RMS
0.6711%; Qwen3.5-0.8B passes 84/84 at 0.2653%. Both retain exact greedy agreement
with their quantized references. Qwen2 host weights occupy 797,246,208 bytes;
Qwen3.5 occupies 1,308,556,544 bytes.

Against the original unquantized Qwen2 reference, all 82 sampled greedy tokens
agree, but median/worst relative logit RMS is 2.4415% / 6.2877%. That report
correctly fails the unquantized numerical threshold: quantization changes the
model. This small teacher-forced trace does not establish task-quality parity.

Qwen2, Qwen3.5, and Llama 3.2 pass mixed-model HTTP switching, residency at
capacities one and three, concurrent requests, reasoning controls, disconnect
recovery, and long-chat checks with Q8 enabled. Shared-storage integration also
passes all 97 unquantized Gemma Apple checks and the 67-test CPU suite.

The first A2 Qwen2 implementation measures only 34.1 / 58.0 / 95.6 tokens/s
at concurrency 1 / 2 / 4 (2048 context, 64 output tokens). It is experimental,
not recommended for speed. Profiling and a matching-precision comparator are
in progress; no cross-engine performance win is claimed. Raw evidence is under
`docs/evidence/quantization/`.

### Q8 grouped-load optimization and comparator

The initial profile attributed 27.7 of 28.1 ms per decoded token to Q8 matrix
kernels. Loading each scale once per explicit 32-value group avoids the costly
expanded scale-load layout. Native fused prefill expansion also removes the
intermediate FP32 weight copies. Narrower launch tiles limit register pressure
for wide matrices. Ten CUDA matrix checks and both full model traces still pass:
Qwen2 worst relative RMS 0.6139%, Qwen3.5 0.2653%.

The optimized Qwen2 A2 HTTP comparison uses the same pinned checkpoint,
Q8_0 projection/output matrices, FP16 input embeddings, FP16 KV, 2048-token
per-request context, 43 prompt tokens, and 64 output tokens. A benchmark-only
GGUF conversion explicitly adds a separately quantized tied output head to
match dllm's storage policy. llama.cpp is pinned to
`836d57176dc699a726c55418e4f96b8ca628e1bf`, native CUDA build for SM86,
flash attention enabled, four slots, prompt caching disabled. Engines run
sequentially; values are medians of five warm rounds.

| Concurrent requests | dllm tokens/s | llama.cpp tokens/s | dllm TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 218.8 | 116.7 | 31.8 | 33.8 |
| 2 | 277.7 | 160.2 | 48.3 | 44.1 |
| 4 | 336.6 | 191.8 | 74.0 | 63.0 |

This is a narrow throughput win, with worse batched first-output latency.
Quantized weights and cache precision match, but kernel arithmetic differs;
generated texts diverge. Independent quantized-reference logits, rather than
cross-engine text equality, provide the numerical gate. Neither these numbers
nor sampled logit agreement establish broad task-quality or engine parity.

## Raw completions and base checkpoints

Issue #12. `/v1/completions` accepts a raw text prompt or explicit token IDs and
emits the text-completion response/stream schema. Shared admission, batching,
sampling, stopping, usage, and cancellation apply to both API routes. Missing
chat templates no longer prevent weight loading; chat rejects those checkpoints
explicitly. Raw IDs are checked against the vocabulary and excluded multimodal
IDs before GPU execution. Prompt batches, multiple completions, echo and logprobs
remain unsupported and rejected.

Live A2 Q8 and M4 unquantized Qwen2 checks both pass with a temporary checkpoint
whose chat metadata is removed: text/token-ID equivalence, stream/nonstream
agreement, exact usage, batch four, seeded sampling, stop sequences, and recovery
after invalid token IDs, oversized input and missing-template chat requests.
The CPU/API suite passes 73 tests.

## Mistral 7B sliding-window qualification

Issue #11. The shared dense executor now recognizes Mistral and applies its
configured sliding window. Official Mistral-7B-Instruct-v0.1 revision
`ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca` passes Q8 A2 mixed-model service at
residency capacities one and three alongside Qwen2 and Qwen3.5. The independent
FP32 eager quantized reference at context 4352 passes 97/97 A2 checks, including
4095/4096/4097 window boundaries, slot reuse and batch reorder/shrink. Worst
relative RMS is 0.24477%, maximum absolute error 0.03812, with exact greedy
agreement. Host weights occupy 7,817,469,952 bytes. This establishes correctness
against Q8 weights, not original-checkpoint task-quality parity.

The initial full-window SDPA allocation exceeded A2 memory. Long masked prefill
now processes query blocks of at most 256 tokens, retaining the exact global
causal/window coordinates while bounding temporary attention scores. CPU checks
compare this with the full mask; Phi-3 retains all 97 passing checks, including
its previous worst relative error. This is bounded attention workspace, not
scheduler-level chunked prefill or a paged KV cache.

## StarCoder2 code models

Issue #13. Official `bigcode/starcoder2-3b` revision
`733247c55e3f73af49ce8e9c7949bf14af205928` uses LayerNorm with bias, biased
projections, GELU MLP, GQA and rotary/window attention. It defaults to tied
embeddings even when that field is omitted. The independent executor shares
native rotary/attention kernels, graph replay and batch slot management; it
does not invoke Transformers during generation. Prefills currently run one
sequence at a time; decode batches normally.

The published weights are FP32. Initial FP16 storage/arithmetic variants failed
strict logit checks despite agreeing on sampled greedy tokens. Unquantized
execution therefore preserves the original FP32 weights, normalization, matrix
arithmetic and KV. Use `--host-gib 16` for this 3B checkpoint. At context 2048,
all 82 checks pass against FP32 eager reference: A2 worst relative RMS 0.02999%,
Apple M4 0.00677%. Synthetic bias/no-bias and tied/untied-head variants also
exercise window boundaries and reordered batches. Real checkpoint qualification
beyond 2048 tokens was subsequently extended: the complete 4352-context trace,
including 4095/4096/4097 window boundaries, passes 97/97 on A2 and M4, with
worst relative RMS 0.00502% / 0.00622%. Apple raw HTTP service and CLI also pass.

The CUDA Q8 path passes all 82 independently quantized-reference checks at
2048, worst relative RMS 0.07770%, with 3,826,790,400 host weight bytes. This
retains FP32 embeddings, biases, normalization, activations and KV; matrix
weights use Q8 blocks. Actual raw code-completion service checks pass, including
text/token-ID equivalence, streaming, batch four, seeded sampling, stopping and
recovery. Q8-vs-original task quality and performance remain separate open
measurements. Original FP32 and Q8 references are both retained as evidence.

## Further Q8 batching work

A fused Q8 Tensor Core path now handles 2–16 FP16 input rows, dequantizing tiles
inside matrix multiplication. FP32 inputs keep the precise vector path; larger
prefill batches retain expanded-weight cuBLAS while tile reuse is investigated.
Fourteen native matrix gates pass, plus 82 Qwen2, 84 Qwen3.5 and 97 Mistral
reference checks. Representative four-row large matrices improve by roughly
1.8–3.1x in isolated kernel timing. This is not an end-to-end speed claim.

The first Mistral cross-engine chat benchmark was rejected as a matched
comparison because templates produced 23 vs 26 input tokens. The benchmark
client can now accept a frozen token-ID array and use raw completions across
engines. The corrected Q8 comparison uses the same 23 input token IDs, 64
output tokens, 2048 context per request, FP16 KV and matching Q8 matrix/FP16
embedding storage on the A2. Five warm rounds give these medians:

| Concurrency | dllm tokens/s | llama.cpp tokens/s | dllm TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 19.63 | 20.74 | 232.8 | 89.5 |
| 2 | 29.45 | 39.81 | 373.4 | 98.5 |
| 4 | 56.53 | 62.03 | 549.2 | 251.1 |

Prompt latency remains a clear gap. Reports are the
`compare-*-mistral-q8-tensorcore-*` files under `evidence/quantization`.

## OLMoE checkpoint qualification

Issue #14. OLMoE has an independent CUDA implementation with whole-projection
Q/K normalization, optional QKV clipping, expert routing, gated expert MLPs and
weighted aggregation. Decode selects expert weight addresses on GPU, including
inside captured graphs. Prefill currently groups tokens with a Python expert
loop; eliminating those synchronization points remains a performance task.
Projection/cache/activation arithmetic is FP32, with FP16 or Q8 matrix storage.
Apple MoE is not yet supported.

Eight routed projection/aggregation cases and sixteen complete synthetic model
checks pass, including repeated and mixed experts, quantized/unquantized weights,
normalized/unnormalized routing, clipping and reordered batches. The official
OLMoE-1B-7B-0924-Instruct revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`
passes all 82 independent Q8-reference checks at context 2048 on A2, including
batch changes and slot reuse. Worst relative RMS is 0.0003882%, with exact
sampled greedy agreement. Host weight storage is 7,448,584,192 bytes. Real mixed
dense/hybrid/MoE service checks pass at batch capacities one and three, including
model switching and concurrency. Reports are `olmoe-q8-replay.json` and
`olmoe-q8-service.json` under `evidence/architectures`.

This qualifies execution against independently quantized weights; original
checkpoint task quality, longer contexts and comparative MoE performance remain
open measurements.

## Q8 short-prefill latency

Fused dequantization/Tensor Core multiplication now handles 2–128 FP16 rows;
larger prompts and FP32 input retain the existing path. A2 measurements select
32/64/128-row tiles for short prefills. Twenty-six native matrix cases and all
82 Qwen2, 84 Qwen3.5 and 97 Mistral reference checks pass.

Repeating the identical-token Mistral comparison above after this change gives:

| Concurrency | dllm tokens/s | llama.cpp tokens/s | dllm TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 20.56 | 20.71 | 84.3 | 89.4 |
| 2 | 31.50 | 39.63 | 150.9 | 100.5 |
| 4 | 59.63 | 61.78 | 309.2 | 143.6 |

Single-request prompt latency improves 2.76x. Batch-two decode and batched
prefill remain gaps; these results do not establish universal engine parity.
Reports use the `mistral-q8-prefill` label in `evidence/quantization`.

## Constrained JSON responses

Issue #15. Optional XGrammar 0.2.8 compiles JSON-object or a bounded JSON-schema
subset. Each request owns its matcher; a model owns a compiler with a 64 MiB
cache limit. CUDA uses XGrammar's Triton mask kernel; Apple uses device bitwise
masking. Masks apply before greedy/top-k/top-p selection and do not modify
shared graph logits. Completed JSON permits EOS; a token-budget cutoff retains
`length` rather than claiming a complete object. No automatic JSON repair runs.

Supported schemas: a single object/array/string/integer/number/boolean/null type;
object properties, required keys and boolean additionalProperties; typed array
items; primitive enum/const; title/description annotations. Unsupported keywords
(including refs, composition, patterns, numeric bounds and cardinality) fail
closed. The compiler uses strict mode and schema property order. Stop strings,
explicit thinking and ignore-EOS conflict with constraints and are rejected.
Qwen3/Qwen3.5 chat constraints automatically disable thinking.

XGrammar and Transformers tokenizer adaptation are optional runtime dependencies;
neither supplies model generation. Grammar compilation and CPU mask generation
add constrained-request overhead; unconstrained sampling retains its fast path.

JSON formatting permits at most eight whitespace characters between grammar
elements, preventing unbounded whitespace-only generation. String contents are
not subject to this formatting limit.

Live service checks pass on A2 Q8 Qwen2.5-0.5B, Qwen3.5-0.8B and Phi-3 mini,
and Apple M4 Qwen2.5-0.5B and Gemma 3-270M. They cover Unicode, strict required
keys/constant values, raw/chat, stream equivalence, mixed constrained/plain
batches, seeded sampling, JSON objects, length truncation, rejection and
disconnect recovery. Tokenizer-level CPU/device mask equality checks cover
Qwen2, Phi-3 and Gemma, including nested JSON and padded vocabulary.
Reports are under `evidence/structured`. No broad schema or tool-call parity
is claimed.

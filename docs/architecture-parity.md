# Architecture qualification beyond the initial decoder matrix

[GitHub issue #3](https://github.com/filthyrake/damendaemon/issues/3) defines the current
parity milestones. Active work targets Qwen3 and newer, prioritizing Qwen3.5.
This document preserves historical qualification evidence, including older
models; its section order is not the current work priority or a task tracker.

## Llama 3.2

Checkpoint: `unsloth/Llama-3.2-1B-Instruct`, immutable revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`. This is an accessible mirror;
qualification applies to that exact checkpoint and its template.

The shared executor now implements standard, linear-scaled, and Llama 3
piecewise rotary frequencies. Dynamic and LongRoPE schedules still fail before loading weights; static
YaRN support and its separate precision policy are qualified below. The frequency implementation has independent
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

| Concurrent requests | damendaemon output tokens/s | vLLM output tokens/s | damendaemon TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 61.40 | 63.24 | 29.28 | 40.97 |
| 2 | 109.33 | 116.67 | 47.25 | 55.44 |
| 4 | 205.03 | 225.74 | 91.04 | 86.72 |

Values are medians; throughput includes prompt processing. This narrow test
shows a 3–9% throughput gap, not general performance parity. Serving environments
use torch 2.6/damendaemon and torch 2.13/vLLM 0.30, respectively. Raw trials are stored
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
eager matches damendaemon at relative RMS 0.000179%, while SDPA differs by 76.6% and
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
CUDA kernels are available; `DAMENDAEMON_GEMMA_WEIGHT_DTYPE=float32` retains full storage.
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
HTTP rounds. This compares two damendaemon implementations, not another engine.
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
reference-only report explicitly has `passed: null` and performs no damendaemon
qualification. This permits sequential GPU reference/runtime validation without
two resident model copies. `replay_reference.py --slot N` can isolate one original
slot for memory-constrained hardware; the report records that reduced scope.

### Phi-3 serving comparison

Mixed-model service passes at GPU residency capacities one and three, with a
12 GiB host weight budget for the larger checkpoint. The stop-sequence API
regression also passes on A2. A matched 2048-context, 64-output-token A2 HTTP
comparison uses the same pinned checkpoint and FP16 weight/KV precision:

| Concurrent requests | damendaemon tokens/s | vLLM tokens/s | damendaemon TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 21.4 | 21.6 | 58.7 | 72.6 |
| 2 | 38.0 | 39.3 | 110.4 | 101.4 |
| 4 | 72.2 | 76.2 | 177.8 | 143.9 |

Medians of five warm rounds. damendaemon's declared FP32 residual/output accumulation
is part of its numerical policy; this does not mean all operations are FP16.
vLLM uses its standard `--dtype half` path, version 0.30 / torch 2.13; damendaemon uses
torch 2.6. vLLM receives a 90% A2 memory budget to fit this checkpoint, and the
engines run sequentially. Throughput is within roughly 1–5% on this workload;
batched first-token latency remains worse. This is not blanket engine parity.

The rendered responses differ in leading whitespace but agree after stripping
outer whitespace. Every measured request has 17 prompt tokens and 64 output
tokens. Independent numerical qualification uses the saved reference traces
described above.

## Experimental Q8_0 weight execution

Issue #10. `DAMENDAEMON_QUANTIZATION=q8_0` quantizes supported safetensors matrices
on host load. CUDA/Triton only; the default remains unquantized. This is not a
GGUF, GPTQ, or AWQ importer. Embeddings, norms, convolution and recurrent state
retain their original storage policy. A tied output head becomes a separate
quantized matrix, so weight savings can be modest on vocabulary-heavy models.
Small-batch decode reads INT8 weights and FP16 scales directly; larger prefill
temporarily expands one matrix at a time. Broad quality evaluation is pending.

The native format matches upstream llama.cpp Q8_0 byte-for-byte in four format
cases. Ten native CUDA matrix cases pass against independently dequantized
matrix multiplication (worst relative RMS 0.00784%). Independent Transformers
references use upstream gguf-py's quantizer, not damendaemon's quantization code.
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
match damendaemon's storage policy. llama.cpp is pinned to
`836d57176dc699a726c55418e4f96b8ca628e1bf`, native CUDA build for SM86,
flash attention enabled, four slots, prompt caching disabled. Engines run
sequentially; values are medians of five warm rounds.

| Concurrent requests | damendaemon tokens/s | llama.cpp tokens/s | damendaemon TTFT ms | llama.cpp TTFT ms |
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

| Concurrency | damendaemon tokens/s | llama.cpp tokens/s | damendaemon TTFT ms | llama.cpp TTFT ms |
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
inside captured graphs. Within the qualified Q8/A2 scope, prefills of one to
four total tokens reuse routed projections; five to 128 use device-counted
grouped projections. Longer prompts retain stable device-side route sorting,
one host count vector per layer, and a Python expert loop. Improving that
long-prompt path remains a performance task.
Projection/cache/activation arithmetic is FP32, with FP16 or Q8 matrix storage.
Apple MoE is not yet supported.

Eight routed projection/aggregation cases and sixteen complete synthetic model
checks pass, including repeated and mixed experts, quantized/unquantized weights,
normalized/unnormalized routing, clipping and reordered batches. The official
OLMoE-1B-7B-0924-Instruct revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`
passes all 82 independent Q8-reference checks at context 2048 on A2, including
batch changes and slot reuse. Worst relative RMS is 0.0003882%, with exact
sampled greedy agreement. Host weight storage is 7,448,584,192 bytes. Real mixed
dense/hybrid/MoE service checks pass at GPU model capacities one and three, including
model switching and concurrency. Reports are `olmoe-q8-replay.json` and
`olmoe-q8-service.json` under `evidence/architectures`.

This qualifies execution against independently quantized weights; original
checkpoint task quality, longer contexts and comparative MoE performance remain
open measurements.


Stable prefill grouping retains identical records on all 82 independent checks
and passes all 24 routed/synthetic GPU checks. On the same resident Q8 checkpoint,
control/candidate/candidate/control measurements produce bitwise-identical
logits throughout. Median prefill times for 32/128/512/1024 input tokens decrease
from 0.4799/0.6143/0.9701/1.4837 seconds to
0.3558/0.4415/0.7964/1.3350 seconds (25.9/28.1/17.9/10.0%). CPU profiles show
1025 versus 49 stream synchronization calls. These are direct prefill timings;
HTTP timing is a separate gate. Reports use the `olmoe-grouped-` prefix under
`evidence/architectures`.

The repeated real HTTP comparison uses the same 512 input IDs, 32 outputs,
Q8 checkpoint, context 2048 and capacity four. Control/candidate/candidate/control
phases each contain five measured rounds after warmup. At concurrency 1/2/4,
median response latency decreases from 1.3077/2.5598/5.0717 seconds to
1.1535/2.2547/4.4885 seconds (11.8/11.9/11.5%). All output text and token counts
match. First-output medians improve by about 15% at concurrency one/four;
concurrency-two control phases vary substantially with admission timing, so its
combined first-output percentage is not treated as an isolated kernel gain.
Raw trials, baseline source hashes and phase medians accompany
`olmoe-grouped-http-summary-a2.json`.

The grouped OLMoE path also passes refreshed real mixed dense/hybrid/MoE service
checks at one and three resident-model capacities: repeat outputs, mixed-model
concurrency, host restoration, thinking controls and disconnect/rejection
recovery all pass (`olmoe-grouped-service-a2.json`).


### OLMoE device-counted grouped projection experiment

An isolated A2 experiment keeps route counts, expert tile assignment and Q8
expert projections on the GPU, using TF32x3 matrix products with FP32 outputs.
It passes all 82 independent whole-model oracle checks (worst relative RMS
0.000242%), but changes reduction order and is not bitwise equivalent. The
initial measurements below compare against the qualified stable grouping above.

Against that improved runtime, alternating same-resident-weight measurements
show prefill medians of 0.175 vs 0.327 seconds at 32 input tokens and 0.325 vs
0.440 seconds at 128. At 512 and 1024 tokens the experiment is slower: 1.018 vs
0.799 and 1.950 vs 1.338 seconds. Boundary checks also win at 1/4/16/64/129
but regress at 256 tokens (0.557 vs 0.533 seconds). These are direct prefill
measurements, not HTTP throughput.

Evidence: [independent oracle](evidence/architectures/olmoe-device-grouped-reference-a2.json),
[dispatch source fingerprint](evidence/architectures/olmoe-device-grouped-reference-dispatch.json),
[prefill comparison](evidence/architectures/olmoe-device-grouped-prefill-a2.json),
and [boundary comparison](evidence/architectures/olmoe-device-grouped-boundary-a2.json).
The first comparison's candidate fingerprint identifies the normal module;
its accompanying dispatch report identifies the actual experimental source.
The boundary report records the experimental source directly.

A second experiment doubles the route tile to 32 rows. All 32 standalone and
82 independent model checks pass. Its measured prefill medians are 0.233,
0.325, 0.483, 0.842 and 1.576 seconds for 32/128/256/512/1024 input tokens;
paired stable-grouping controls are 0.331/0.453/0.528/0.800/1.341 seconds.
It improves the 256-token point but still regresses at longer prompts and
loses to the 16-row experiment for short prompts. It remains experimental.
See [larger-tile measurements](evidence/architectures/olmoe-device-grouped32-prefill-a2.json)
and [independent model gate](evidence/architectures/olmoe-device-grouped32-reference-a2.json).

The normal runtime now selects the 16-row tile only for the tested A2 Q8
OLMoE geometry, context at most 2048, batch capacity at most four and prompts
of 1–128 tokens. Longer prompts and other configurations retain the previous
path. The integrated runtime passes all 82 independent checks including mixed
short/long prompts and reused decode slots (worst relative RMS 0.000389%);
349 local tests pass, including selection boundaries. Mixed-model lifecycle
qualification also passes with Qwen2.5, Qwen3.5 hybrid and OLMoE at GPU model
capacities one and three: repeated responses, host restoration, mixed-model
concurrency, thinking controls and disconnect/rejection recovery. See
[service proof](evidence/architectures/olmoe-device-grouped-service-a2.json),
[normal-runtime oracle](evidence/architectures/olmoe-device-grouped-normal-reference-a2.json)
and [source fingerprints](evidence/architectures/olmoe-device-grouped-normal-source.json).

The normal streaming server also passes repeated control/candidate/candidate/control
measurements with 32 identical input IDs and 32 output tokens. Median total
latency falls from 0.634/1.209/2.382 seconds to 0.460/0.861/1.668 seconds at
concurrency 1/2/4 (27.4/28.8/30.0%). Aggregate output throughput including prefill
rises from 50.44/52.70/53.46 to 69.46/73.89/76.58 tokens/s. Responses and token
counts are identical throughout. Single-request first output improves from
0.361 to 0.188 seconds; C4 improves from 1.449 to 0.751 seconds. C2 first-output
control medians vary between phases (0.719 vs 0.540 seconds), so its aggregate
59% reduction includes admission timing variation. These are warm short-prompt
service results, not decode-only throughput or a cross-runtime comparison.
See [HTTP measurements and phase medians](evidence/architectures/olmoe-device-grouped-http-summary-a2.json).

A further 32-row/128-column projection tile passes all 32 standalone and 82
independent full-model checks, but does not improve on the narrower experimental
tile. Prefill medians are 0.245/0.340/0.505/0.884/1.657 seconds at
32/128/256/512/1024 input tokens, versus paired stable controls of
0.342/0.456/0.536/0.803/1.343 seconds. It stays disabled; see
[wider-tile timings](evidence/architectures/olmoe-device-grouped32x128-prefill-a2.json).

An isolated packed-prefill experiment combines concurrent short prompts for
matrix and expert work while retaining separate attention slices and cache slots.
A derived oracle regroups unchanged independent prompt targets and teacher-forced
decode continuations: six cases cover mixed lengths, repeated prompts, two to
four requests, reordered slots and a 130-total-token fallback. The sequential
control and experimental implementation both pass all 234 checks (experimental
worst relative RMS 0.000098%). This is independent target reuse, not a new
reference computed by damendaemon. Normal-runtime qualification is described below.
See [oracle provenance](evidence/architectures/olmoe-short-batch-reference-provenance.json),
[control](evidence/architectures/olmoe-short-batch-control-reference-a2.json),
[candidate](evidence/architectures/olmoe-packed-prefill-reference-a2.json)
and [dispatch](evidence/architectures/olmoe-packed-prefill-reference-dispatch.json).

Alternating same-resident-weight measurements against the already-improved
sequential path reduce prefill for four 32-token prompts from 0.705 to 0.323
seconds, a mixed 31/65 pair from 0.407 to 0.278 seconds, a 64/64 pair from
0.439 to 0.322 seconds and a 31/31/65 triple from 0.582 to 0.324 seconds.
The 65/65 pair falls back and measures 0.465 seconds for both variants.
Tiny and uneven boundary cases also improve, including four single-token
prompts (0.155 to 0.045 seconds) and a 1/127 pair (0.361 to 0.318 seconds).
All measured greedy outputs agree and complete logits remain inside the
existing numerical gates. These are direct prefill timings, not HTTP claims.
See [batch measurements](evidence/architectures/olmoe-packed-prefill-timing-a2.json)
and [boundary measurements](evidence/architectures/olmoe-packed-prefill-boundary-a2.json).

The normal OLMoE runtime now packs two to four requests when the combined
prompt length is at most 128 tokens within the already-qualified A2 Q8 scope.
It shares the existing forward implementation, keeps attention/cache segments
separate and retains the sequential path outside that bound. Both independent
suites pass on the integrated code: 234 regrouped checks and the original 82
mixed-length checks. All 350 local tests pass, including selection boundaries.
Refreshed mixed-model lifecycle proof also passes with Qwen2.5, Qwen3.5 hybrid
and OLMoE at one and three GPU-resident models, including host restores, mixed
concurrency and disconnect/rejection recovery. See
[service evidence](evidence/architectures/olmoe-packed-service-a2.json), [normal batch gate](evidence/architectures/olmoe-packed-normal-batch-reference-a2.json),
[original model gate](evidence/architectures/olmoe-packed-normal-full-reference-a2.json)
and [source fingerprints](evidence/architectures/olmoe-packed-normal-source.json).

Repeated default-scheduler HTTP measurements (32 input IDs, 32 outputs,
control/candidate/candidate/control) preserve every response and token count.
C4 median total latency improves from 1.667 to 1.384 seconds (17.0%), first
output from 0.750 to 0.468 seconds (37.6%) and aggregate output throughput
including prefill from 76.60 to 92.21 tokens/s. C1 and C2 total latency remain
effectively unchanged (0.463 vs 0.461 and 0.861 vs 0.860 seconds). C2 candidate
first-output phase medians vary (0.285 vs 0.221 seconds), reflecting request
admission; do not interpret its combined first-output statistic as a uniform
kernel gain. No scheduler coalescing setting was changed. See
[HTTP measurements](evidence/architectures/olmoe-packed-http-summary-a2.json).

An isolated offset-prefill experiment passes the original 82 independent model
checks with 128-token chunks, including the 2032-token prompt and subsequent
slot reuse/decode. Worst relative RMS is 0.000198%. This initial isolated gate
precedes the normal-runtime and prefix/service qualification below.
See [chunked oracle](evidence/architectures/olmoe-chunk128-reference-a2.json)
and [dispatch provenance](evidence/architectures/olmoe-chunk128-reference-dispatch.json).

The same isolated offset implementation also passes 48 independently computed
CPU FP32 eager-reference checks for bounded prefix reuse. These cover exact
hits, divergent suffixes, shorter prefix restoration, slot/decode isolation,
eviction and the 512-token snapshot budget; worst relative RMS is 0.000110%.
The snapshots remain copied per-executor KV state, not shared pages. See
[prefix numerical proof](evidence/architectures/olmoe-prefix-experiment-a2.json)
and [experimental dispatch](evidence/architectures/olmoe-prefix-experiment-dispatch.json).

The shared normal OLMoE forward pass now accepts offset attention segments and
can skip final logits on nonfinal chunks. Opt-in chunking/prefix reuse is exposed
only within the existing A2 Q8 geometry/context/batch scope. The integrated
127-token chunk path passes all 82 independent model checks, and the packed
prompt regression passes all 234 checks. All 351 local tests pass, including
invalid continuation offsets and unchanged cache lengths on rejection. The normal
prefix reference passes all 48 CPU FP32 eager-reference checks (worst relative
RMS 0.003459%), and 256-token chunks pass all 82 original model checks (worst
relative RMS 0.000582%). Actual HTTP prefix qualification passes below;
interleaving measurements are recorded below.
See [normal prefix proof](evidence/architectures/olmoe-prefix-normal-a2.json) and
[256-token chunk proof](evidence/architectures/olmoe-chunk256-normal-reference-a2.json).
See [normal chunk gate](evidence/architectures/olmoe-chunk127-normal-reference-a2.json),
[packed regression](evidence/architectures/olmoe-chunk-normal-batch-regression-a2.json)
and [source fingerprints](evidence/architectures/olmoe-chunk-normal-source.json).

The actual prefix service passes exact/partial hits, concurrent response equality,
stream-disconnect recovery and model-alias switch isolation with a 16-GiB host
budget. At 1536 input tokens and eight outputs, five warm trials per setting
measure median server first-token time of 1792.2 ms without caching and 6.1 ms
for an exact cache hit; median complete HTTP latency is 1901.0 vs 110.9 ms.
A divergent three-token suffix measures 1822.3 vs 72.9 ms to first token.
The cache uses a 4096-token per-executor budget and copied KV snapshots. These
are cache-hit comparisons in two sequential phases, not uncached throughput
or a cross-runtime comparison. Outputs, input usage and cache-hit telemetry
match expectations, and model switches clear the old executor's cache.
See [live prefix service proof](evidence/architectures/olmoe-prefix-service-a2.json).

Actual active-stream interleaving preserves the same greedy outputs across full
prefill and 128/256-token chunks. Three trials per setting introduce a 1536-token
prompt while a seven-token prompt is streaming 64 output tokens; the long request
emits eight tokens. Median results are:

| Chunk tokens | Maximum active-stream gap | Long request | Short request |
| --- | --- | --- | --- |
| Full prefill | 1798 ms | 1.932 s | 2.554 s |
| 128 | 318 ms | 3.669 s | 4.183 s |
| 256 | 504 ms | 3.042 s | 3.611 s |

Chunking smooths the active stream but increases both total request durations in
this workload. Full prefill remains the default; chunking is an explicit latency
tradeoff, not a general throughput improvement. See
[stream arrivals and measurements](evidence/architectures/olmoe-chunk-interleaving-a2.json).

### OLMoE tiny-prefill routed projections

Reusing the existing routed decode projections during prefill avoids mostly
empty grouped matrix tiles for tiny prompts. Same-resident alternating trials
compare that strategy with grouped Q8 prefill, with unchanged attention and KV
handling. All measured outputs agree greedily and remain within the existing
numerical limits. Varied corpus prompts improve through 16 tokens, but a
concentrated repeated-token workload regresses after four tokens:

| Tokens | Varied grouped / routed ms | Repeated-token grouped / routed ms |
|---|---:|---:|
| 1 | 39.122 / 20.436 | 39.136 / 21.253 |
| 2 | 61.552 / 24.061 | 40.053 / 24.136 |
| 3 | not measured | 41.030 / 28.828 |
| 4 | 92.610 / 34.774 | 41.936 / 34.752 |
| 7 | not measured | 57.631 / 65.127 |
| 8 | 139.319 / 70.250 | 57.695 / 70.172 |
| 15 | not measured | 66.865 / 114.006 |
| 16 | 173.033 / 119.066 | 66.806 / 119.115 |

The normal runtime uses a conservative cutoff of four total tokens, including
packed prompts, within the existing qualified Q8/A2 scope. Both the isolated
candidate and actual normal runtime pass 182 independent FP32 HF/GGUF checks
through context 2048, including scalar 1/2/4/8/16-token prompts, packed totals on
both sides of the boundary, reordered slots, and subsequent decoding. Worst
relative RMS is 3.883e-6 and every greedy token agrees. The local regression
suite passes 351 tests. [Independent reference](evidence/architectures/olmoe-q8-tiny-reference-a2.json),
[normal replay](evidence/architectures/olmoe-q8-tiny-normal-reference-a2.json),
and [source fingerprints](evidence/architectures/olmoe-q8-tiny-normal-source.json)
record qualification. [Varied prompts](evidence/architectures/olmoe-q8-routed-prefill-a2.json)
and [concentrated prompts](evidence/architectures/olmoe-q8-routed-prefill-concentrated-a2.json)
retain the gains and regressions; these are prefill-only comparisons, not HTTP
throughput measurements.

The normal service also passes exact/divergent prefix reuse, three-token suffix
execution, concurrent response equality, disconnect recovery, and model alias
switching with host restoration. The live test compares cached and uncached
execution; it is not a matched before/after latency benchmark for this change.
See [live prefix and residency proof](evidence/architectures/olmoe-q8-tiny-prefix-service-a2.json).

The real HTTP control/candidate/candidate/control comparison uses one identical
input token, 32 outputs, context 2048, batch capacity four, the default scheduler,
and five measured rounds per phase/concurrency after an excluded warmup. Every
response and usage record agrees. Pooled medians are:

| Concurrency | Old / new total seconds | Old / new first-output ms | Old / new aggregate output tokens/s |
|---|---:|---:|---:|
| 1 | 0.325749 / 0.311226 | 45.187 / 29.311 | 98.157 / 102.720 |
| 2 | 0.575578 / 0.551968 | 72.773 / 50.411 | 110.094 / 115.351 |
| 4 | 0.995591 / 0.980277 | 96.353 / 74.412 | 127.718 / 129.327 |

These modest whole-request gains are much smaller than the isolated prefill
speedup. The C1 control phase medians vary from 0.314 to 0.359 seconds, while
candidate phases are 0.307 and 0.313; the pooled reduction is not a uniform
per-phase speedup. Throughput includes prefill and HTTP overhead, not decode
alone. [Summary, phase medians and reproduction commands](evidence/architectures/olmoe-q8-tiny-http-a2.json)
link the corresponding retained raw trials and source identities.

### OLMoE short-prefill graph experiment (not enabled)

The qualified Q8 grouped path can be captured and replayed with changed input
IDs and exact outputs. A bounded two-entry single-prompt graph experiment passes
changed prompts/routes, slot isolation, subsequent decode, 129-token fallback,
eviction, graph release, and model unload checks. Packed and chunked requests
bypass this cache; unsupported candidate scoring still rejects before mutation.

Alternating complete-prefill measurements, including input staging, show only
67.056 to 65.924 ms at 16 tokens and 248.938 to 247.826 ms at 128 tokens. A first
cached call including warmup/capture costs about 445 ms and 1,177 ms respectively.
These small warmed gains do not justify normal dispatch or a default change.
The isolated prototype and its evidence remain available for future tuning;
normal OLMoE prefill graph eligibility is unchanged. See
[capture probe](evidence/architectures/olmoe-q8-prefill-graph-probe-a2.json),
[lifecycle and alternating measurements](evidence/architectures/olmoe-q8-prefill-graph-lifecycle-a2.json),
and [experimental dispatch](evidence/architectures/olmoe-q8-prefill-graph-lifecycle-dispatch.json).

### OLMoE four-bit experiment (not registered)

Routed Q4_0 expert kernels pass 24 checks against upstream GGUF-produced bytes
and independently dequantized FP32 values. The initial whole-model experiment
keeps tiny router matrices and embeddings in FP16, other projections in Q4_0,
and arithmetic/norms in FP32. Stored host weights occupy 4,043,595,776 bytes,
compared with 7,448,584,192 bytes for the qualified Q8 configuration; this is
storage accounting, not a peak-memory or quality claim.

The first independent 512-context whole-model gate fails one of 82 checks.
At decode position 498 in slot 1, relative RMS is 4.232% and maximum absolute
logit error 0.912, despite greedy agreement. The preceding 496-token prefill
already differs by 0.187% RMS. Q4 OLMoE remains disabled in normal registration;
the existing error thresholds are unchanged. A graph-disabled replay reproduces
the same failure and magnitude, excluding graph replay as the cause of this
observed discrepancy. Expert routing and weight equivalence still require
diagnosis before performance claims or promotion.
See [kernel proof](evidence/architectures/moe-q4-routed-kernels-a2.json),
[failed full-model gate](evidence/architectures/olmoe-q4-model512-a2.json)
and [explicit storage policy/dispatch](evidence/architectures/olmoe-q4-model512-dispatch.json).

A separate grouped Q4 prefill kernel dequantizes tiles directly instead of
expanding complete expert matrices. It passes 48 independent cases covering
shared and route-specific inputs, concentrated and mixed routing, ragged output
dimensions, and batches 1/4/17/65 (worst relative RMS 3.065e-7). This isolated
kernel gate does not qualify full-model execution or establish a performance
gain. See [grouped kernel evidence](evidence/architectures/moe-q4-grouped-kernels-a2.json)
and [graph-disabled model failure](evidence/architectures/olmoe-q4-model512-eager-a2.json).

An exact comparison of all 3,219 coefficient tensors found five differences
between the initial reference and runtime: embeddings and four layer-zero expert
projections. The initial reference omitted the runtime's FP16 storage rounding
before Q4 conversion. Very small checkpoint values therefore produced different
coefficients (maximum projection difference 1.449e-5). The independent reference
now explicitly rounds embeddings and projection inputs to FP16, while continuing
to use upstream GGUF quantization/dequantization. This corrects the reference's
storage policy; it does not change runtime arithmetic or numerical thresholds.
The original failure remains preserved. See
[exact coefficient mismatch evidence](evidence/architectures/olmoe-q4-initial-weight-mismatch-a2.json).

The corrected reference matches all 3,219 tensors exactly, but reproduces the
4.232% decode discrepancy. Instrumentation observes the first expert-membership
split at layer zero, prompt row 239: experts 49 and 27 exchange eighth/ninth
place with probability gaps 1.304e-8 (native) and 9.313e-9 (reference). Router
input RMS differs by 9.179e-7. FP64 router projections on each captured input
retain their respective ordering, locating the initial difference before the
router multiplication. Later expert changes amplify the discrepancy. No
tie-handling heuristic or relaxed threshold has been introduced. See
[aligned coefficient and route diagnosis](evidence/architectures/olmoe-q4-aligned-diagnostic-a2.json)
and [diagnostic dispatch](evidence/architectures/olmoe-q4-aligned-diagnostic-dispatch.json).

Two further controls replay the original retained trace: explicit FP32 eager
prefill attention and grouped Q4 expert prefill each reproduce the same single
decode failure (approximately 4.232% RMS), with all greedy tokens agreeing.
Neither control is promoted. See [eager attention](evidence/architectures/olmoe-q4-eager-attention-a2.json)
and [grouped Q4 replay](evidence/architectures/olmoe-q4-grouped-replay-a2.json).

Same-resident alternating control/candidate/candidate/control prefill trials
show a short-prompt benefit and long-prompt regression. All measured outputs
agree greedily, with worst control/candidate RMS 6.543e-7. These are comparisons
between two experimental Q4 implementations, not independent qualification or
HTTP throughput results. Warmups and profiling are excluded from timings.

| Prompt tokens | Original Q4 prefill seconds | Grouped Q4 seconds |
|---|---:|---:|
| 1 | 0.066390 | 0.047909 |
| 4 | 0.177307 | 0.114942 |
| 16 | 0.324656 | 0.204374 |
| 32 | 0.365194 | 0.211311 |
| 128 | 0.439237 | 0.394514 |
| 256 | 0.526651 | 0.674340 |
| 512 | 0.792316 | 1.241064 |

Thus the grouped prototype is not a universal replacement; normal Q4 OLMoE
remains disabled. [Raw alternating trials and synchronization profiles](evidence/architectures/olmoe-q4-grouped-prefill-a2.json)
and [dispatch/source identity](evidence/architectures/olmoe-q4-grouped-prefill-dispatch.json)
retain both gains and regressions.

An additional FP64 normalization control, retaining FP32 output and other
arithmetic, reproduces the same single failure (4.232% RMS). It is diagnostic
only and is not promoted. [Replay](evidence/architectures/olmoe-q4-fp64-norm-a2.json)
and [control identity](evidence/architectures/olmoe-q4-fp64-norm-dispatch.json)
preserve this negative result. Issue #23's open question is resolved by the
router-boundary diagnosis (#577) below.

### Q4 OLMoE router-boundary diagnosis (#577)

The near-tie flip mechanism is now bounded by two measurements. The
[boundary-frequency study](evidence/architectures/olmoe-q4-router-boundary-study.json)
(CPU reference, all 27,760 layer/position rows of the 512-token trace) finds
the top-9 cutoff probability gap below 1e-6 for only 11 rows (396.25 per
million rows), minimum 5.588e-9, median 1.09e-3: the observed flip is a rare
boundary crossing, not one of many near-misses. The
[origin localization](evidence/architectures/olmoe-q4-origin-localization-a2.json)
(A2, guarded, layer-0 stage capture on the 496-token prompt) shows the
divergence originates in the QKV projection — the first stage above 1e-7
(5.906e-7 relative RMS) — and stays near 1e-6 through attention into the
router input (9.179e-7, matching the aligned diagnostic) before the route
flip amplifies it to 5.067e-3 at the layer-0 MLP output. Embedding is exact
and the native attention path is internally exact; no stage diverges beyond
fp32 reduction-order noise between the CUDA and CPU arithmetic stacks.

Verdict and router-boundary policy: the Q4 path is numerically sound; the
retained 4.232% failure is a genuine fp32-amplified near-tie, not a defect.
Q4 OLMoE stays disabled in normal registration. MoE route boundaries carry no
tolerance across independent arithmetic stacks: a sub-ppm relative difference
in router input can flip an expert selection wherever the cutoff gap is
comparable, so greedy single-token agreement is not a qualification gate for
paths with near-tie routers. See the
[router-boundary README](evidence/architectures/olmoe-q4-router-boundary-README.md).


## Q8 short-prefill latency

Fused dequantization/Tensor Core multiplication now handles 2–128 FP16 rows;
larger prompts and FP32 input retain the existing path. A2 measurements select
32/64/128-row tiles for short prefills. Twenty-six native matrix cases and all
82 Qwen2, 84 Qwen3.5 and 97 Mistral reference checks pass.

Repeating the identical-token Mistral comparison above after this change gives:

| Concurrency | damendaemon tokens/s | llama.cpp tokens/s | damendaemon TTFT ms | llama.cpp TTFT ms |
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

## Qwen3 accuracy recovery

Issue #2. Layerwise replay reproduces the two Qwen3-0.6B failures at lengths 263
and 1028. Residual error starts at zero in the embedding and accumulates through
middle layers to roughly 2–3%; this is not just output-head rounding. Increasing
only Q/K/V arithmetic precision did not clear the gate.

`DAMENDAEMON_QWEN3_PRECISION=float32` now enables FP32 matrix/activation/KV arithmetic
while retaining FP16 matrix storage. The pinned BF16 checkpoint's values are
exactly representable in this storage. All 84 checks pass against the independent
FP32 reference, worst relative RMS 0.002080%. Since #2's resolution (2026-10-21)
this path is the CUDA DEFAULT for unpacked Qwen3 checkpoints
(`qwen3_default_precision`); quantized weights keep the `standard` fp16 default
their own gates qualified, and `DAMENDAEMON_QWEN3_PRECISION=standard` remains
an explicit opt-out that reproduces the historical two-position failure
(1.19%/1.41%) on a fresh trace.

Initial A2 HTTP medians for the accurate path at concurrency 1/2/4 are
108.33 / 172.96 / 214.32 tokens/sec, TTFT 49.0 / 76.7 / 140.3 ms. Four-request
throughput trails the prior standard path (325.7 tokens/sec), motivating further
matrix-kernel work. The FP32 vLLM run gives 66.23 / 92.86 / 174.26 tokens/sec and
50.1 / 75.1 / 108.1 ms. Both use the pinned checkpoint, 26 input/64 output tokens,
2048 context and FP32 arithmetic/KV; damendaemon stores exact checkpoint matrix values
in FP16, whereas this vLLM run stores FP32 weights. This is explicitly a storage
policy difference, not a matched-storage engine comparison. Reports and the
layerwise diagnostic are under `evidence/models`.

### Compensated Tensor Core optimization

The accurate CUDA Qwen3 mode now uses scaled high/residual FP16 components for
3–128-row matrix inputs. Each component is multiplied by the stored FP16
weights; products and sums accumulate in FP32. This approximates FP32 arithmetic
rather than asserting bitwise equivalence to IEEE FP32 matrix multiplication.
Input/output tensors, normalization, residuals and KV remain FP32. One/two-row
work keeps native FP32 GEMV; large prefills use the prior FP32 matrix path.

All 35 matrix gates (zero, small, ordinary and very large finite activations),
84 Qwen3-0.6B and 84 Qwen3-1.7B reference checks pass. The plain tensor fallback
also passes all 84 small-model checks. The unchanged 1%/0.5 logit limits and exact
greedy checks remain the qualification criteria.

A2 small-model medians at concurrency 1/2/4 are now 107.23 / 170.68 / 271.50
tokens/sec, with TTFT 45.5 / 74.2 / 125.8 ms. Four-request throughput improves
26.7% over the first accurate path; it still trails the faster, numerically
provisional standard path. The accurate path remains opt-in. Reports use the
`qwen3-small-staged` label in `evidence/models`.

## Large-model hot-swap lifetime fix

Issue #16. The scheduler now releases its completed group executor reference
before entering the next model session. Previously its worker-local variable
kept the old weights/KV alive while the residency cache attempted eviction and
the replacement model checked free VRAM. Individually fitting models could thus
fail to swap when their combined footprints exceeded the device.

A lifecycle regression test and actual A2 Q8 OLMoE → Mistral → OLMoE requests
pass with one GPU-resident model. The two measured individual process footprints
sum to more than the A2's total VRAM; the final request restores from host and
matches the first response. No disk weights are deleted. Full response/timing/
memory evidence is in `evidence/architectures/large-model-swap.json`.

## Dynamic model management

Issue #17. The server can start empty and accept local snapshot registration,
explicit warm loading, GPU/host unloading and residency status through HTTP and
CLI. CLI repository resolution/download completes before registration. Aliases
are process-local and cannot be replaced with different paths; repeated
registration and unloading are idempotent. Checkpoint files are never removed.

Management uses the scheduler's bounded queue and acts as a FIFO barrier:
active generations drain before controls run, and later generations cannot
extend that group past a waiting control. Registration validates architecture,
context, tokenizer metadata, weight headers and host capacity before publishing
an alias. A failed registration leaves the registry unchanged. Generation after
unload restores the registered snapshot normally.

The error path also detaches cross-thread exception tracebacks from GPU worker
frames. Retaining a client exception can no longer retain the old executor while
a replacement loads; readable worker stack text is preserved as exception notes.
A regression test reproduced this failure before the fix. Unit coverage includes
control fairness, admission saturation, atomic registration, state transitions,
HTTP field validation and retained-exception lifecycle behavior.

Live A2 and M4 tests pass empty startup, idempotent add, warm load, GPU-only
unload, full host/GPU unload, loading after eviction, unload during an active
stream, invalid registration/load recovery, model switching and all four CLI
commands. Reports are under `evidence/management`. The registry does not persist
across process restarts; startup aliases remain supported.

The accurate Qwen3 path also passes the real mixed Qwen2/Qwen3/SmolLM2 service
checks at GPU-residency capacities one and three, and the Qwen2 numerical batch
regression. Reports are `qwen3-small-accurate-service.json` and
`qwen2-precision-regression.json` under `evidence/models`.

## Sentence embeddings

Issue #18. The independent BERT encoder qualifies
`sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It implements absolute position/type
embeddings, bidirectional attention, post-layer normalization, exact GELU,
masked mean pooling, and L2 normalization. Checkpoint metadata must declare the
supported Transformer/Pooling/Normalize pipeline. No Transformers model runs
in production; the FP32 eager Transformers model is an independent validator.

CUDA uses FP16 dense matrices with FP32 embeddings, normalization, GELU and
residuals. An initial all-FP16 implementation failed the unchanged batch
invariance gate on a whitespace input. The mixed-precision implementation
passes reference and batch-invariance gates: relative RMS below 0.002, maximum
absolute difference below 0.001, cosine similarity above 0.99999. Apple uses
FP32 throughout and passes the same gates. Tests cover mixed lengths, reordered
batches, Unicode, the 256-token boundary, normalization and a basic semantic
ordering check; this is not a retrieval-quality benchmark.

Live HTTP checks on A2 and M4 pass text/token input equivalence, exact
float/base64 equivalence, indexed batch order, usage accounting, concurrent
requests, encoder/decoder switching with host restoration, and recovery after
invalid task/input requests. The unit suite has 94 passing tests.

Warm encoder medians, 30 measured iterations after five warmups, including
input upload and output download but excluding tokenization and HTTP:

| Device | Input tokens | Batch | Latency ms | Inputs/s |
|---|---:|---:|---:|---:|
| A2 | 32 | 1 | 3.06 | 327 |
| A2 | 32 | 32 | 5.17 | 6192 |
| A2 | 128 | 32 | 18.36 | 1743 |
| A2 | 256 | 32 | 38.77 | 825 |
| M4 | 32 | 1 | 3.96 | 253 |
| M4 | 32 | 8 | 4.57 | 1750 |
| M4 | 256 | 32 | 98.21 | 326 |

Batched input uploads reduced the A2 32-input/32-token median from 6.93 to
5.17 ms in sequential before/after runs. No competing-runtime performance
claim is made. CUDA and Apple arithmetic and PyTorch versions differ. Full
measurements, reference checks, and API checks are in `evidence/embeddings/`;
`scripts/benchmark_encoder.py` reproduces the warmed workload.

Remaining limits: no request coalescing, no shortening
of vectors, quantized encoders or arbitrary SentenceTransformers pooling modules.
Trained BERT reranking (#35) and T5 encoder-decoder generation (#29) are now
qualified separately; their task contracts do not change the embedding pipeline. Each HTTP request executes
its own input batch; generation groups drain before an embedding request.
Register the encoder with `quantization: "none"` if the server default is Q8.

CUDA graph follow-up: bounded LRU reuse (four batch/length shapes, batches up to
8) reduces 32-token input latency from 3.06 to **0.67 ms** at batch1, and from
3.66 to **1.54 ms** at batch8 (1498 and 5187 inputs/s respectively). At 256 tokens,
batch1 improves from 3.26 to **1.66 ms**; larger compute-bound batches change
little. First use of a shape pays warmup/capture cost; these are warm medians,
not cold request latency. `--no-graph` disables reuse. The expanded numerical
proof also exercises replay with changed data, graph cache eviction, and
retained output tensors. Live HTTP/model-switching checks pass with graphs.
Request coalescing remains unimplemented. See `minilm-a2-graphs-*.json`.


## Token control qualification

Issue #19. Presence/frequency/repetition penalties and logit bias pass nine
independent scalar-oracle cases on CUDA and MPS, including negative logits,
negative penalties and repeated history. Real Qwen2.5-0.5B checks compare eight
successive token choices to independent scalar arithmetic in the serial
runtime and the HTTP batched runtime. Chat/raw/SSE routes, mixed default/biased
concurrent requests, and invalid-vocabulary recovery pass on both devices.
99 unit tests pass. No request may mutate reusable graph logits.

Matched A2 default-request regression check against commit `23fc857`, same
Qwen2.5-0.5B revision, FP16 weights, context256, 43 prompt tokens, 64 output
tokens and five measured rounds after warmup:

| Concurrency | Before tokens/s | After tokens/s | Before TTFT ms | After TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 138.56 | 137.97 | 26.85 | 26.58 |
| 2 | 234.20 | 235.24 | 40.92 | 40.49 |
| 4 | 432.54 | 444.16 | 63.58 | 61.53 |

Throughput changes range from -0.4% to +2.7% in this sequential A/B run; this
supports no material default-path regression, not a speedup claim. The optional
penalty path uses sparse device tensor updates and additional launches; its
performance under long histories/high concurrency remains an optimization
follow-up. Raw evidence is in `evidence/sampling/`.


## Per-model quantization residency

Issues #10 and #18. Registration now fixes quantization per alias; memory
estimation and loading use that explicit policy. Re-registering an existing
alias with a different policy is rejected. Unloading and reloading retain it.
The environment supplies only the server-startup default, avoiding global
policy mutation during mixed requests.

A2 live proof switches between a Q8 Qwen2.5-0.5B decoder and an unquantized
MiniLM encoder with one GPU-resident model and two host-resident models. Both
restore from host with identical results and correct reported precision;
encoder text/token/base64, concurrent requests, alias-conflict rejection, and
invalid-task recovery pass. The 101-test unit suite covers independent alias
policies and policy retention through full unload/reload. See
`evidence/embeddings/minilm-a2-q8-mixed-service.json`.

## GPT-2 learned positions

Issue #20. `openai-community/gpt2` at immutable revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` passes **84/84** independent-reference
checks on both A2 and Apple M4. This adds learned absolute position embeddings,
transposed Conv1D checkpoint projections, LayerNorm, tanh GELU and tied output
weights to the independent runtime. FP32 weights, arithmetic and KV are used.
The existing CUDA/Metal small-batch matrix and attention kernels and residency
scheduler are shared; model execution is not delegated to Transformers.

The tests cover heterogeneous prompt lengths, reordered/shrinking decode
batches, reused cache slots, raw Unicode tokenizer equivalence, and the final
learned position at context1024. Attempts to decode beyond the position table
are rejected without changing cache lengths. Maximum relative RMS logit error
is 0.00034994 on A2 and 0.00010385 on M4; all greedy tokens agree. Gates remain
relative RMS below 0.01 and maximum absolute error below 0.5. The reference is
FP32 eager Transformers (CUDA for the A2 test; CPU for the M4 test).

The first CUDA batched test exposed noncontiguous query rows from packed QKV
slices; making those rows contiguous fixed the native attention input contract.
This is covered by the heterogeneous/reordered batch checks. Live raw API
text/token equivalence, streaming, four concurrent requests, seeded sampling,
stops, and error recovery pass on both devices. The unit suite has 103 tests.

This qualifies the pinned 124M base checkpoint and its 1024-position limit.
It has no chat template. Cross-attention, alternate attention scaling/upcast
modes, untied output heads, GPT-2 quantization, and other GPT architectures are
rejected or unqualified. Evidence is in `evidence/architectures/gpt2-*.json`.

GPT-2 matched FP32 A2 HTTP comparison, same pinned checkpoint and token IDs,
context1024, 14 prompt tokens and 64 output tokens, five measured rounds:

| Concurrent requests | damendaemon tokens/s | vLLM tokens/s | damendaemon TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 271.00 | 236.06 | 12.43 | 30.63 |
| 2 | 476.50 | 424.90 | 20.23 | 43.62 |
| 4 | 753.68 | 800.57 | 42.16 | 48.70 |

Both use FP32 weights/arithmetic; environments differ (damendaemon torch2.6, vLLM0.30
with torch2.13). These medians include prompt processing and exclude first
model-load warmup. This narrow result leads at concurrency1/2 but remains about
6% behind at concurrency4; it does not establish general runtime parity.
The no-graph, pure-Torch CUDA fallback also passes all 84 recorded reference
checks. Full evidence: `compare-*-gpt2-fp32-*.json` and
`gpt2-a2-torch-reference.json` in `evidence/architectures/`.

## Interleaved prompt chunks

Issue #21. The opt-in `--prefill-chunk-size` scheduler path bounds prompt tokens
processed between active decode steps. Partially prepared prompts reserve cache
slots; output sampling waits for the final chunk. Cancellation discards the
partial request, and slot reuse starts again at position 0. Other-model and
management requests remain FIFO barriers. Unsupported architectures retain
full prefill. This is scheduling, not paged KV or shared-prefix reuse.

Offset-aware attention includes previous cached chunks and respects sliding
windows. CUDA FP16 non-windowed attention uses a lower-right causal bias to
retain fused Flash SDPA; materializing the offset mask had disabled that fast
path. Intermediate chunks skip the unused vocabulary output projection.

The final path passes 82 Qwen2 A2 reference checks, 84 accurate-Qwen3 A2 checks,
82 Llama 3.2 A2 checks, and 82 Qwen2 M4 checks with uneven 127-token chunks.
Q8 Mistral passes 97 checks using 257-token chunks through its 4096-token window. Qwen3 qualification uses
`DAMENDAEMON_QWEN3_PRECISION=float32`; the existing standard-Qwen3 accuracy limitation
is unchanged. Phi-3's 257-token-chunk test fails one absolute-error gate despite
matching every greedy token, so it is deliberately excluded from chunking;
its full-prefill support is unchanged. Failed evidence is retained alongside
passing evidence, rather than widening the tolerance.

Live HTTP experiment: a 1536-token prompt requesting 8 output tokens joins after
four content events from a short request with a 64-token output limit. Three
warmed trials per setting, same greedy outputs throughout:

| Device | Chunk tokens | Median largest stream gap ms | Short request seconds | Long request seconds |
|---|---:|---:|---:|---:|
| A2 | disabled | 111.60 | 0.572 | 0.170 |
| A2 | 128 | 33.75 | 0.744 | 0.408 |
| A2 | 256 | 31.17 | 0.597 | 0.226 |
| M4 | disabled | 590.26 | 2.159 | 0.825 |
| M4 | 128 | 120.18 | 2.534 | 1.345 |
| M4 | 256 | 158.97 | 2.243 | 1.023 |

Gaps measure SSE content arrival, including transport, not internal GPU token
latency. This workload shows a responsiveness/total-latency tradeoff and does
not justify changing the global default. The initial slower implementation's
measurements and the optimized run are both retained in `evidence/chunked/`;
`scripts/benchmark_chunked.py` reproduces the workload. The 108-test unit suite
includes decode/chunk ordering, cancellation without output, reused slots,
management barriers, budget validation and unsupported-architecture fallback.

## Bounded prefix reuse

Issue #22. `--prefix-cache-tokens` enables executor-owned GPU KV snapshots on
the decoders eligible for chunked prefill. At most four entries and the selected
total token budget are retained. An exact prompt can also reuse its saved final
logits; a divergent or shorter prompt resumes from a matching prefix, leaving
its final token to be recomputed when no exact logits exist. KV is copied into
independent request slots. This is neither paged KV nor zero-copy sharing.

Caches disappear with GPU model eviction, do not cross aliases/precisions, and
use weak executor references. Snapshots are skipped when the memory reserve
cannot be maintained; partial allocations are discarded on allocator failure.
Prompt usage remains unchanged, while `cached_prompt_tokens` reports actual
reuse. The default remains disabled because cache misses add copying/memory
costs and this first implementation retains whole bounded snapshots.

Independent full-prompt FP32 eager references pass 48 checks each for Qwen2 on
A2/M4 and accurate Qwen3 on A2: exact matches, divergent/shorter prefixes,
post-snapshot decoding, reordered/reused slots, token/entry bounds and eviction.
Worst relative RMS is 0.0084753 for Qwen2 CUDA, 0.0000130 for Qwen2 M4, and
0.00000252 for accurate Qwen3 CUDA. The existing 1% RMS/0.5 absolute/exact-greedy
gates are unchanged. 114 unit tests pass, including low-memory and failed
allocation behavior and executor-lifetime isolation.

Live Qwen2.5-0.5B HTTP tests compare disabled caching with a 4096-token budget,
a 1536-token repeated prefix and eight output tokens. Exact timings are medians
of five warm requests; partial timings use five distinct three-token suffixes,
each reusing 1536 tokens but requiring new model computation:

| Device | Cache | Exact first-token ms | Exact HTTP total ms | Partial first-token ms |
|---|---|---:|---:|---:|
| A2 | disabled | 103.48 | 162.86 | 113.32 |
| A2 | enabled | 1.79 | 60.61 | 25.34 |
| M4 | disabled | 559.61 | 838.81 | 568.41 |
| M4 | enabled | 3.81 | 282.73 | 42.69 |

First-token values come from runtime telemetry; HTTP totals are measured by
the client. This is repeated-prefix latency evidence, not a general throughput
comparison. Greedy outputs agree with uncached requests. Four concurrent
requests, model alias switching/restoration, divergent suffixes, and combined
256-token chunking plus caching pass. An initial first-partial-hit CUDA test
exposed a one-second lazy PyTorch attention import; optional offset-attention
support now initializes before the server advertises readiness. That startup
cost is moved, not eliminated. Both the original result and final evidence are
retained in `evidence/prefix/`.

## Native Q4_0 weight execution

Issue #23. Packed projection rows use two four-bit values per byte and one FP16
signed scale per 32 values; embeddings/norms retain their original execution
precision. The CUDA path reads nibbles directly for decode and short-prefill
matrix products; longer prefill temporarily expands one projection at a time.
The latter workspace cost still applies despite smaller resident weights.

Four independent upstream gguf-py format cases are bit-exact. Twenty-six A2
matrix cases pass against independently dequantized values (worst relative RMS
0.0003833). Qwen2.5-0.5B passes its 2048-context model gate against independently
Q4-quantized Transformers weights, including reordered/reused slots. These
checks establish execution correctness for the quantized weights, not fidelity
to the unquantized model. The full 116-test suite passes.

The separate CPU FP32 drift script uses FP16 weight conversion followed by
independent upstream Q4/Q8 dequantization and ten published authored passages:
171 teacher-forced tokens, original perplexity 17.71, Q8 17.50, Q4 26.89;
next-token agreement 95.32% for Q8 and 74.27% for Q4. A tiny corpus can move
perplexity in either direction and cannot establish general task quality.
The input corpus and raw results are in `evidence/quantization/`.

The format reference is llama.cpp/gguf-py at
`836d57176dc699a726c55418e4f96b8ca628e1bf`. The optional comparator generator now
accepts `--format q4_0`; it starts from unquantized same-checkpoint GGUF weights,
retains unquantized embeddings and quantizes an independent output head.
Broader model and matched serving qualification are still in progress. No
four-bit throughput improvement is claimed from kernel or CPU checks alone.

The Q4 model matrix now also passes Qwen3-0.6B in accurate FP32 arithmetic mode
(84 checks, worst relative RMS 0.00001317) and Llama3.2-1B in FP16 (82 checks,
worst RMS 0.007198), both at 2048 context. Qwen2's 82 checks have worst RMS
0.007822. Live HTTP tests pass disk/host/GPU restoration with one and three
resident models, mixed concurrent aliases, four-request same-model batching,
reasoning-template controls, streaming disconnect recovery and long prompts.
These results qualify execution of independently quantized weights; they do
not replace the original-weight drift study or pending Mistral timing results.

## Static YaRN rotary scaling

Issue #5. Dense Qwen2/Qwen3/Llama/Mistral configuration now accepts static YaRN
frequency interpolation and attention scaling, including explicit original
context, fast/slow rotation bounds and attention factor. The method is described
by [Peng et al., YaRN](https://arxiv.org/abs/2309.00071). The serving implementation
is independent; Transformers is used only for validation. Dynamic NTK,
LongRoPE, partial-head YaRN and custom mscale variants remain unsupported.

Twenty-one table cases on CPU and M4 cover head dimensions 64/96/128, standard,
linear, Llama3 and four YaRN configurations, with positions through 131071.
Both frequencies and attention-scaled sine/cosine tables match Transformers.
These table checks alone are not a 128K model qualification.

The full execution fixture uses unchanged pinned Qwen2.5-0.5B weights with a
separate configuration: original context 512, maximum 2048, YaRN factor 4.
`scripts/make_yarn_fixture.py` reproduces it without modifying source weights.
This deliberately exercises the scaling boundary with small memory; it is not
a pretrained long-context checkpoint or a long-context quality claim.

A2 and M4 each pass 82/82 independent CPU FP32 eager-reference checks, in both
full prefill and 127-token chunks, with greedy agreement, RMS below 1%, and
absolute error below 0.5. Worst RMS is 0.00001745 on A2 and 0.00001680 on M4.
A2 YaRN uses FP32 arithmetic/KV and FP16 matrices through the existing split
Tensor Core/native GEMV path. M4 retains its existing FP32 execution policy.

The initial CUDA FP16 path failed the unchanged gate: worst RMS 0.01019 against
FP32 and 0.01357 against matched FP16 eager execution. Both failed reports are
retained alongside successful evidence in `evidence/rotary/`. That path is not
enabled for YaRN. The accuracy policy has memory/performance costs; no YaRN
throughput parity or trained long-context quality claim is made here.

Mistral7B Q4 now passes all 97 independent CPU FP32 eager-reference checks at
4352 context, including the 4095/4096/4097 sliding-window boundary; worst RMS
0.003294, host weight storage 4,262,273,024 bytes. The original implementation's
matched Q4 serving comparison is poor and retained as the optimization baseline:

| Concurrent requests | damendaemon tokens/s | llama.cpp tokens/s | damendaemon TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 4.18 | 34.20 | 165.90 | 67.24 |
| 2 | 14.99 | 64.09 | 370.07 | 86.85 |
| 4 | 29.68 | 84.02 | 662.97 | 138.79 |

Identical pinned Mistral checkpoint, matching Q4_0 projection/FP16 embedding
policy, 23 explicit prompt token IDs, 64 output tokens, context 2048, FP16 KV,
A2 only, sequential servers and five warm measured rounds. This establishes a
real execution bottleneck; Q4 is not yet a performance improvement in damendaemon.

## GPT-NeoX / Pythia

Issue #24. Independent execution now supports GPT-NeoX head-interleaved packed
QKV weights, partial-head rotary embeddings, biased LayerNorm/projections,
exact GELU and parallel or sequential residual branches. The first qualified
checkpoint is `EleutherAI/pythia-160m` revision
`50f5173d932e8e61f858120bcb800b97af589f46`, with parallel residuals and 25% rotary
dimensions. Checkpoint tensors are validated and reordered into the runtime's
whole-Q/K/V layout. Quantization and nonstandard rotary schedules are rejected.
Execution uses FP32 weights/arithmetic/KV; this base model has no chat template.

On Apple M4, all 82 independent CPU FP32 eager-reference checks pass through
2048 context, including reused/reordered/shrinking slots (worst relative RMS
0.00001517, maximum absolute error 0.01184). Raw text/token-ID completions,
streaming, concurrent requests, sampling/stops and invalid-request recovery pass.
The 119-test suite includes geometry rejection and explicit QKV row mapping.
A2 also passes 82/82 checks against the same reference (worst relative RMS
0.00002888), and its live raw API suite passes.

Matched M4 HTTP comparison: identical checkpoint converted to FP32 GGUF,
identical explicit prompt IDs, FP32 KV, 2048 context, 64 output tokens, five
warm rounds, sequential engines:

| Concurrent requests | damendaemon tokens/s | llama.cpp tokens/s | damendaemon TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 116.20 | 129.99 | 20.94 | 9.68 |
| 2 | 211.84 | 197.00 | 43.69 | 13.25 |
| 4 | 377.98 | 267.98 | 88.79 | 38.73 |

This is a narrow throughput/latency tradeoff, not general parity. GPU prefill is
still per-prompt; GPT-NeoX chunked prefill/prefix reuse are not yet enabled.
Evidence is stored under `evidence/architectures/pythia160m*`.

Q4 optimization replaces repeated per-element scale loads with block loads and
paired low/high-nibble Tensor Core products for FP16 decode/short prefill. The
profiled original 14336-wide down-projection GEMV used 255 registers and spilled
352 values; its kernel took about 4.0 ms. The paired product takes about 0.43 ms
with no reported spills on the same matrix. An intermediate blockwise GEMV and
split-reduction tile sweep are retained as evidence; split GEMV was not selected.

All 26 CUDA matrix gates pass and all 97 Mistral model checks still pass
(worst RMS 0.002870). The same serving workload improves to 20.06/37.70/70.39
output tokens/s at concurrency 1/2/4, with TTFT 81.90/190.06/325.04 ms. This is
4.80x/2.51x/2.37x faster than the original Q4 path, while still behind the matched
llama.cpp result, especially for one request. The full 119-test suite passes.
Further native decode work is in progress; this remains an explicit measured
gap rather than a claim of general four-bit performance parity.

Matched A2 FP32 Pythia-160M comparison, using the same explicit prompt IDs,
2048 context, 64 output tokens and five warm rounds:

| Concurrent requests | damendaemon tokens/s | vLLM tokens/s | damendaemon TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 256.08 | 240.87 | 15.16 | 24.24 |
| 2 | 436.95 | 440.62 | 23.72 | 31.47 |
| 4 | 695.17 | 807.41 | 57.43 | 43.49 |

The four-request throughput deficit is about 14%; this architecture is
numerically/API qualified but is not a general performance-parity claim.
Serving environments remain torch 2.6/damendaemon versus torch 2.13/vLLM 0.30.

An optional independent CUDA C++ kernel now vectorizes Q4 nibble conversion
with FP16 pairs while preserving the dequantized FP16 rounding boundary and
FP32 accumulation. The first scalar CUDA attempt was slower; its source and
measurements are retained. Eighteen vectorized matrix shapes pass with and
without bias (36 checks); full Mistral replay passes 97/97, worst RMS 0.003693.
The C interface validates dimensions, uses the caller's stream and reports
launch errors. Python validates device/dtype/shape/contiguity and library ABI.
The library is explicitly selected at startup with `DAMENDAEMON_Q4_CUDA_LIBRARY`.

Measured native selection is restricted to one FP16 input row, input width at
least 4096 and at least 1024 outputs. Batched and smaller matrix workloads stay
on the faster Tensor Core path. Matched Mistral serving reaches
22.53/37.89/70.64 tokens/s at concurrency 1/2/4, with TTFT
81.43/142.44/320.10 ms. Single-request throughput improves another 12.3%; batch
throughput is effectively unchanged. The remaining gap to llama.cpp is explicit.
The native build used CUDA 12.6 for SM86; other device/toolkit builds are not
qualified by this evidence.

The native selection also passes Llama3.2-1B Q4 replay, 82/82 checks at 2048
context, worst RMS 0.007112. The full unit suite now has 121 passing tests,
including native ABI mismatch rejection and explicit opt-in/device boundaries.

GPT-NeoX now fuses CUDA partial rotary and KV writes through the shared slotted
kernel, preserving unrotated tail values. All 115 kernel checks pass, including
nine partial-width/head-dimension combinations with unused cache entries filled
with NaNs; full-head normalized rotary remains covered. The 82-check Pythia
model replay remains unchanged numerically. A2 throughput improves to
274.87/459.25/742.04 tokens/s, TTFT 13.00/27.84/47.34 ms, at concurrency 1/2/4.
The four-request gap to the same vLLM baseline falls to about 8%. MPS retains
its previously qualified path. Raw measurements are preserved alongside the
initial baseline, rather than replacing it.

## BLOOM / ALiBi

Issue #26. Pinned [BigScience BLOOM-560m](https://huggingface.co/bigscience/bloom-560m)
revision `ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971` adds an independent
ALiBi decoder. The attention-bias method follows
[Press et al.](https://arxiv.org/abs/2108.12409); Transformers' BLOOM implementation
is used only as the independent test oracle. damendaemon implements geometric head
slopes, head-interleaved QKV reordering, embedding and block LayerNorm,
tanh GELU, residual connections and tied output weights. CUDA decode adds
ALiBi inside the shared slotted attention kernel. Prefill uses bounded
128-query attention tiles; it does not implement scheduler chunking for BLOOM.

Both A2 and M4 pass 84/84 checks through 2048 context, including 2047-prefill
plus final decode, head/slot permutations and cache reuse. Independent reference:
CPU FP32 eager Transformers, PyTorch 2.13. Production uses FP32 matrices,
arithmetic and KV. Worst relative RMS: A2 0.000007259, M4 0.000004620.
A2 maximum absolute logit error is 0.003022, comfortably inside the existing
0.5 gate. All selected tokens match. Eighteen separate CUDA attention checks
cover head counts 1/12/16, dimensions 64/96/128, FP16/FP32, reordered slots and
NaN-filled unused cache; the existing kernel regression suite also passes.

The checkpoint has no chat template. Raw text/token-ID equality, SSE,
four-request batching, seeded sampling, stop handling and rejection recovery
pass on A2 and Apple M4. Quantization, prefix caching and interleaved prefill remain
unsupported. The default 2048 context when absent from the legacy configuration
is a conservative qualification boundary, not a claim of length extrapolation.

Initial BLOOM M4 FP32 comparison: same pinned checkpoint and explicit token IDs,
FP32 KV, 64 output tokens and five measured warm rounds. **This initial comparison
is not context-matched:** damendaemon used 2048 slots, but llama.cpp silently capped its
slots at 1024 based on converter metadata. Corrected measurements follow separately. The
llama.cpp build is the existing pinned comparator; the GGUF is an FP32 conversion
of the same safetensors checkpoint. Throughput includes prompt processing.

| Concurrent requests | Initial damendaemon tokens/s | Native Metal attention | Packed prefill + native attention | llama.cpp tokens/s | Final damendaemon TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 16.89 | 36.43 | 37.07 | 36.29 | 32.92 | 34.01 |
| 2 | 26.23 | 68.84 | 70.49 | 57.22 | 42.61 | 41.95 |
| 4 | 36.93 | 123.42 | 120.56 | 82.15 | 102.30 | 58.64 |

The initial path copied entire KV slots into general tensor attention at each
decode step. Native Metal attention reads only live positions and applies ALiBi
inside the kernel. It passes nine dedicated ALiBi cases and all 29 existing
Metal kernel checks. Native and subsequently packed-prefill versions each pass
84/84 independent BLOOM checks. The final live completion-service proof passes.

Packed prefill shares projections across prompts while keeping attention and
cache slots separate, bounds total temporary tokens to one configured context,
and computes the vocabulary projection only for each sequence's last row.
Prefill attention consumes current K/V views directly, avoiding full-cache copies.
This cuts four-request TTFT from the native-attention-only 253.84 ms to 102.30 ms,
but still trails llama.cpp's 58.64 ms. These initial observations do not establish matched performance parity.
The packed-prefill A2 recheck passes 84/84 independent checks (worst relative
RMS 0.000008169), and its live completion-service proof also passes.

Corrected, **context-matched** BLOOM comparisons supersede the initial
llama.cpp numbers above. Both servers' startup logs confirm four slots with
2048 tokens each. The explicit GGUF metadata override is
`bloom.context_length=int:2048`; no weights or tokenizer were modified. The
converter's default used hidden size when `seq_length` was absent. Both benchmark
launchers now query `/props` and fail before timing if actual slot context differs
from the requested value, with a regression test for the observed 1024/2048 case.

Same pinned FP32 weights, FP32 KV, ten explicit prompt token IDs, 64 output
tokens, five warm measured rounds, isolated GPU runs:

| GPU | Concurrent requests | damendaemon tokens/s | llama.cpp tokens/s | damendaemon TTFT ms | llama.cpp TTFT ms |
|---|---:|---:|---:|---:|---:|
| M4 | 1 | 37.07 | 36.50 | 32.92 | 34.44 |
| M4 | 2 | 70.49 | 57.13 | 42.61 | 42.55 |
| M4 | 4 | 120.56 | 82.55 | 102.30 | 58.79 |
| A2 | 1 | 69.13 | 46.84 | 28.14 | 25.62 |
| A2 | 2 | 121.93 | 74.15 | 36.55 | 33.25 |
| A2 | 4 | 213.81 | 123.63 | 79.84 | 90.07 |

These results establish useful throughput for this checkpoint/workload, while
preserving the remaining M4 four-request first-token latency deficit. A2 packed
prefill improves TTFT from 33.40/61.12/130.05 ms to 28.14/36.55/79.84 ms. The
A2 vLLM 0.30 FP32 baseline never became ready: its engine stayed GPU-busy with
no new progress after warmup for roughly ten minutes. Only that owned process
group was stopped, and its startup log/command/incomplete-result record are
preserved. There is **no valid vLLM BLOOM throughput result**.

The llama.cpp source pin is `836d57176dc699a726c55418e4f96b8ca628e1bf`.
The M4 binary reports build 11381 at that commit; the A2 archive-built binary
reports commit unknown, so its provenance relies on the existing pinned source
archive/build record. Comparator commands and raw trials are committed. The A30
workload was left running; all NVIDIA inference work selected the A2 UUID.

## T5 encoder-decoder (issue #29)

FLAN-T5-small adds independently implemented text encoder-decoder inference on
A2 and M4: bidirectional relative encoder attention, causal relative decoder
attention, cached cross-attention and gated GELU. Separate context accounting
allows a full encoder input plus decoder output. CUDA uses native attention and
graph replay; Apple uses native projection, norm and relative decoder-attention kernels.

[Qualification, limitations and initial timing evidence](evidence/t5/README.md)
include 198 independent FP32 logit checks per GPU plus real API and residency
proof. Original T5-small additionally qualifies tied embeddings and ReLU, with
573 checks through the full decoder context on both GPUs. The llama.cpp server
baseline fails startup or matched output validation; a separate public-API core
comparison produces identical token IDs and exposes damendaemon's slower encoder but
faster decoder. Core timings and HTTP throughput are reported separately.

## Mamba-1 recurrent state (issue #30)

Independent Mamba-130m text generation adds causal convolution and selective
state-space recurrence, with native CUDA/Metal kernels and fixed per-slot state.
Both A2 and M4 pass independent FP32 logits/state checks through 2048 context and
real HTTP/residency proof. [Evidence and matched three-engine measurements](evidence/mamba/README.md)
show A2 throughput competitive with vLLM for this checkpoint; Apple is still
behind llama.cpp. Both 130M and 370M checkpoints are now qualified, including CUDA Q8 matrix
weights. Other Mamba generations/hybrids remain unsupported.
[Q8 storage, numerical limits and measurements](evidence/mamba/q8/README.md).

Mamba chunked prefill and complete-prefix recurrent snapshots now work on both
GPUs, including CUDA Q8. No rewind to shorter/divergent prefixes is permitted.
[Qualification and latency tradeoffs](evidence/mamba/chunks/README.md): A2
128-token chunks reduce the measured active-stream gap from 162 to 23 ms,
while increasing incoming request time from 206 to 278 ms.

## Mamba-2 (issue #31)

Pinned original 130M weights are independently loaded and executed on CUDA/Metal
after lossless offline FP32 safetensors preparation. Both GPUs pass 139 reference
checks plus state, chunk/prefix and real HTTP/residency qualification through
2048 context. [Evidence and measured gaps](evidence/mamba2/README.md).
Single-group pure Mamba-2 only; no claim for Codestral or attention hybrids.

Mamba-1/2 now support native Metal Q8 matrix projections as well as CUDA Q8.
FP32 activations/state remain unchanged; quantization is opt-in and measured
model drift is nonzero. [Q8 qualification and performance gaps](evidence/mamba2/q8/README.md).

## Function tools and direct decisions (issues #32–#35)

Qualified Qwen2/Qwen3 templates now support auto/none tool calls, assistant/tool
history and grammar-constrained required/named single calls, including correlated
`anyOf` function schemas. [GPU/API evidence](evidence/tools/README.md). Automatic
strict selection, additional tool formats and multiple forced calls remain open.

Unquantized Qwen2/Qwen3 expose direct candidate-token scoring on A2/M4, without
answer decoding. Optional candidate-keyed CUDA graphs reduce repeated-shape
latency while retaining explicit capture/memory costs. Scores are conditional
on the provided tokens and uncalibrated. [Evidence](evidence/scoring/README.md).
This is independent work inspired by [NInfer/Jev research](upstream-inference-research.md),
not support for their model weights or a dependency on their runtimes.

The pinned MS MARCO MiniLM-L6-v2 BERT cross-encoder now supports trained scalar
relevance scoring via `/v1/rerank`. A2/M4 FP32 execution matches an independent
CPU reference through512-token query/document pairs; task switching and HTTP
ranking semantics are qualified. [Reranker evidence](evidence/reranking/README.md).
These raw relevance scores are also not calibrated confidence.

Opt-in bounded CUDA prefill graphs separately improve repeated short Qwen2/Qwen3
requests, with verified model unload/eviction and matched HTTP measurements.
[Capture costs, memory and warm latency](evidence/prefill-graphs/README.md).

## Multilingual XLM-RoBERTa relevance

Issue #36 adds independent BGE reranker v2 M3 execution with padding-aware
absolute positions and its trained scalar classifier. The pinned checkpoint
passes14 CPU FP32 reference cases per device on A2 and M4 through2048 tokens,
including multilingual text, internal padding tokens, long length buckets and
reversed document order. HTTP lifecycle and concurrency checks pass on both.
[Exact revision, measured latency, arithmetic gates and scope](evidence/reranking/xlm-roberta/README.md).
The model's8192-token capacity is distinct from the tested2048-token limit;
quantized reranking and other classifier variants remain unqualified.

## GPT-OSS-20B packed raw execution

Issue #39. Checkpoint `openai/gpt-oss-20b` at
`6cee5e81ee83917806bbde320786a8fb61efebee` now has an independent native CUDA
executor with packed MXFP4 experts, GPU-resident top-k routing, biased
projections, clipped/interleaved SwiGLU, learned attention sinks, alternating
local/full attention and fractional-boundary YaRN. It does not load an upstream
inference runtime. Floating checkpoint values are preserved during loading;
FP16 matrix storage is used only when lossless. Residuals, KV and reductions are FP32; larger dense projections use the
existing compensated Tensor Core path. Prepared host weights occupy 13,775,206,656 bytes.

On A2, 72 checks pass against an independent Transformers 5.18 CPU FP32 eager
reference with one layer expanded at a time. These include all 24 layer outputs,
13- and 137-token causal prompts, graph decode from 5- and 129-token prefixes,
exact greedy logit agreement, isolation of another live slot, and slot reuse.
Worst relative RMS error is 0.0006742%; the established gates are 1% relative RMS,
0.5 maximum absolute error and exact greedy agreement. The executor context is
256 with two slots; this is not a 128K or GPT-OSS-120B qualification.

Raw HTTP completions pass streaming/non-streaming equivalence, disconnect
recovery, two simultaneous requests, actual GPU swapping to Qwen2 and back,
graph/executor reclamation, and explicit unload/host restoration. Response
metadata identifies native `mxfp4` weights rather than mislabeling them as
unquantized. At this initial stage Harmony chat/tools were unqualified and the
chat route was rejected; subsequent text-chat qualification is recorded below.
Raw text continuation is not a chat-quality assessment.

Initial A2 measurements: approximately 90.4 ms per graph decode step (~11 tokens/s)
for a short raw prompt; 10.98 seconds for a 129-token prefill. The latter is a
known bottleneck: the initial routed expert implementation rereads weights per
token. A warm 5-token prompt / 16-output-token HTTP request takes 1.759 seconds;
cold disk-to-response takes 15.53 seconds and host restoration plus response
6.93 seconds in the service probe. These are narrow initial measurements,
not a matched comparison with another engine. Two-slot reference peak allocated
memory is 13,930,619,904 bytes. Supporting reports and reproduction scripts are
linked in [GPT-OSS evidence](evidence/gpt-oss/).

### Grouped GPT-OSS prefill

The following warmed HTTP measurements supersede the initial routed-prefill
numbers above. Prompts of at least 16 tokens now group token rows by expert.
Only one expert projection is expanded at a time. FP16 expansion is selected
only when every value is exactly representable, verified from the checkpoint
scales; otherwise expansion remains FP32. Compensated Tensor Core projections
retain FP32 activation/residual/KV/output policy. Decode keeps packed GPU routing.
Expert token counts use power-of-two buckets to bound compilation variants;
extra rows are discarded before route combination. This is still ordinary
monolithic prefill, not interleaved prompt scheduling or a graph-captured router.

A2, same pinned checkpoint/resident server, context 256, two slots, eight output
tokens, four alternating measured rounds after warmup. Every response and usage
count matches between the two modes:

| Prompt tokens | Routed HTTP seconds | Grouped HTTP seconds | Routed / grouped first-token seconds |
|---|---:|---:|---:|
| 16 | 1.859 | 1.271 | 1.220 / 0.631 |
| 32 | 3.059 | 1.409 | 2.419 / 0.770 |
| 129 | 10.375 | 1.601 | 9.732 / 0.959 |
| 224 | 17.531 | 1.734 | 16.885 / 1.092 |

Complete requests improve 1.46–10.11× on these workloads. First-use compilation
is excluded. Peak allocated memory in the largest grouped workload is
13,959,017,984 bytes. All 72 independent CPU reference checks pass with grouped
prefill and bounded token-count buckets. `DAMENDAEMON_GPT_OSS_PREFILL=routed` retains the
baseline for diagnosis. This comparison changes expert execution arithmetic
from FP32 reductions to compensated products; it is not a bitwise-logit claim,
and it is not a comparison with another engine.

### GPT-OSS packed decode reductions

The default routed kernel now reduces 512 input columns at a time for widths
of at least 2048, using four output rows for shared gate inputs and eight for
per-route down inputs. The original full-width path remains available internally
for controlled comparisons. A2 compiler measurements show 188–189 registers
for the full-width tile versus 80/128 for the selected smaller tiles, with zero
spills in either case. Narrowing byte offsets did not provide consistent gains
and was not retained. The actual first-layer packed gate projection improves
from 2.217 ms to 1.440 ms; its preselected expanded FP32 baseline is 1.645 ms.

All exhaustive format/activation checks, real checkpoint projection checks and
72 full-model CPU reference checks pass after changing the reduction order.
Out-of-range int64 expert IDs remain masked, including values that would wrap
into valid indices if narrowed. CPU/HTTP regression suite: 222 passing tests.

Matched A2 complete HTTP requests, same resident model, grouped prefill,
32 output tokens, four alternating warmed rounds. Each path has its own CUDA
capture; both graph sets share the same weights/KV. Responses and usage match:

| Prompt tokens | Concurrent requests | Full reduction seconds | Smaller reduction seconds |
|---|---:|---:|---:|
| 5 | 1 | 3.228 | 2.149 |
| 32 | 1 | 3.547 | 2.614 |
| 129 | 1 | 3.766 | 2.835 |
| 32 | 2 | 6.659 | 4.813 |

Seconds measure wall time until all requests complete. Single-request decode
increases from approximately 10.9–11.0 to 16.3 tokens/s; this excludes prompt
processing. Full-request latency decreases 25–33% in these cases. This is an
internal matched A/B, not vLLM/llama.cpp parity. Evidence and individual samples:
[decode qualification](evidence/gpt-oss/decode/). Reproduce with
`scripts/probe_mxfp4_chunks.py`, `scripts/replay_gpt_oss.py`, and
`scripts/benchmark_gpt_oss_decode.py`.

### GPT-OSS text Harmony chat

Text chat is now qualified separately from the earlier raw-completion gates.
The native token-framed parser recognizes checkpoint control IDs, separates
`analysis` into `reasoning_content`, and emits final/commentary text as `content`.
It never treats header strings as answer text. Alias spellings are validated
against the tokenizer because this pin uses `<|channel|>` / `<|return|>` where
some Harmony examples use older names. At this text-only qualification stage function handoffs and non-text control
frames failed explicitly; function-call qualification is recorded below.

Nineteen CPU checks match exact prompt token IDs with Transformers 5.18 and
streamed text channels with the official `openai-harmony` 0.0.8 Rust parser,
including captured output from the actual native model. The reference library
is used only in validation; production keeps its existing tokenizer and native
executor. The [official format repository](https://github.com/openai/harmony)
and the pinned checkpoint template are the protocol sources.

Real A2 HTTP checks at 512 context/two slots pass ordinary and Unicode answers,
prior-turn recall, reuse of the returned assistant message (including its
reasoning field), SSE/non-stream equivalence, token caps inside headers and
analysis, final-text stopping, disconnect during analysis, concurrent channel
isolation, GPU graph reclamation on model swap, and equivalent host restoration.
Generated answer examples include `4`, `Ada.`, `café 🐈`, and `5` after continuing
a returned message. These are protocol/smoke examples, not a model-quality suite.

`reasoning_effort=low|medium|high` is GPT-OSS-only; the template default remains
medium. CLI `--reasoning-effort` uses the same validation and prints final text.
All generated control/body tokens count toward the completion limit and usage.
Stop strings match content, not analysis. At this stage GPT-OSS tool calls were
unsupported; JSON constraints, chat logprobs and chat `ignore_eos` remain unsupported. Raw completion behavior
is preserved. See [Harmony evidence](evidence/gpt-oss/harmony/); 231 CPU/HTTP
regressions pass alongside the hardware and independent-format checks.


### GPT-OSS Harmony function handoffs

The independent token parser now recognizes declared `functions.NAME` recipients
on the commentary channel, buffers arguments, and publishes one structured call
only after the handoff marker and complete JSON validation. Duplicate keys,
nonfinite values, non-object arguments, unknown functions and malformed framing
fail explicitly. Incomplete arguments at the completion limit are suppressed,
with `finish_reason=length`; no partial call is published. Function bodies never
become answer text. Literal Qwen-style envelopes in ordinary Harmony content
remain literal text, even when tools are enabled.

Actual A2 HTTP proof at context 1024/two slots produced `weather(city="Paris")`,
returned `finish_reason=tool_calls`, then consumed a supplied result and answered
with 19°C. No external function was executed by the runtime or this test.
Streaming/nonstream content, reasoning, arguments and usage matched; call IDs
were independently generated. Header/argument caps, `none` choice, concurrent
requests and disconnect recovery passed. Twelve CPU checks independently matched
HF Transformers 5.18 prompt IDs (including returned reasoning/tool-result history)
and the official Rust Harmony parser 0.0.8 (including captured generated tokens,
Unicode arguments and alternate header ordering). Neither is a serving dependency.

History adapts JSON argument strings to objects, null content to empty text, and
`reasoning_content` to the checkpoint template's `thinking` field. The pinned
template assumes one function per assistant message; extra calls are rejected
rather than silently dropped. History containing both commentary content and
reasoning is explicitly unsupported by this template integration. Auto/none only:
forced/named/strict calls, built-in Python/browser tools, multimodal control frames
and structured chat constraints remain unsupported. Tool schemas guide generation;
auto calls are checked for valid JSON, not full schema conformance.

Scripts: `verify_harmony_tools.py`, `verify_harmony_tool_format.py`, and the
updated `verify_harmony_service.py`. Evidence is in
[Harmony function-call checks](evidence/gpt-oss/harmony-tools/). The local CPU/HTTP
suite passes 235 tests. These are protocol/correctness checks, not a comparative
throughput or model-quality benchmark.


### Direct scalar CUDA graph dispatch

Direct `Runtime.generate`/CLI calls now reuse the batch executor's captured
one-row graph instead of bypassing it through the disabled base scalar graph.
The existing HTTP scheduler path is unchanged. On the A2, matched six-round
32-step decode trials show 2.52–3.02× gains for Qwen3-0.6B with explicit FP32
policy and about 1.02× for native MXFP4 GPT-OSS-20B. All logits and free-running
tokens match exactly in these comparisons. Independent scalar reference replays
pass 25 Qwen3 checks and 72 GPT-OSS checks; 237 CPU/HTTP regressions pass.
No-graph/non-CUDA fallback remains unchanged, as do existing Gemma/paged scalar
overrides. See [measurements and methodology](evidence/scalar-decode/README.md)
and GitHub issue #40. These gains exclude load/prefill and are not HTTP speedups.

### Exact normal-scale expert decode

GPT-OSS now selects an exact bit-construction path when a checkpoint projection's
scales all lie in [2,252], retaining generic MXFP4 handling otherwise. All 4,016
normal code/scale values match independent reference bits, and all 72 model
reference checks pass. Matched A2 production HTTP trials improve single-request
decode about 2.2–2.4% (16.22–16.29 to 16.58–16.67 tokens/s), with 1.2–2.4% lower
wall time in the recorded run. This is a small optimization, not a new precision
policy. 238 CPU/HTTP tests pass. An alternative compensated Tensor Core route
layout was measured and rejected as slower; it is excluded from runtime selection.
See [full proof, limits and rejected experiment](evidence/gpt-oss/normal-scales/README.md).

### GPT-OSS longer-context memory and open numerical gate

A 32 MiB prefill attention workspace bound and shorter expert-intermediate
lifetimes allow a two-slot, 4096-context replay on A2. Attention partitioning is
bit-identical; the measured attention allocation peak drops from 1.08 GiB to
95.8 MiB, including the full output. The original nearly tied expert choice
produced a greedy mismatch at a 789-token prefix with both bounded and unbounded
attention. IEEE FP32 attention projections now repair all five targeted output
prefixes; longer-context internal-state qualification remains open.
See [all measurements, failed checks and reproduction](evidence/gpt-oss/context-4k/README.md)
and [issue #41](https://github.com/filthyrake/damendaemon/issues/41).

### GPT-OSS bounded fused prefill attention

Prefills of 256 or more tokens now use bounded fused FP32 sink attention and
IEEE FP32 attention projections. The sink normalization comes from the fused
operator's log-sum-exp output. Full attention partitions heads; windowed
attention trims each query tile's key range. The unchanged decode graphs and
short-prefill path remain in use. `DAMENDAEMON_GPT_OSS_ATTENTION=native` provides the
bounded native attention path with the same IEEE projection policy for matched
comparisons.

Two independent 72-check reference sets pass, as do all five sensitive output
prefixes and real 4K raw/tool HTTP lifecycle tests. Long-context absolute
hidden-state failures are retained in issue #41, including the observed
variation between independent CPU and CUDA references. See the
[fused-prefill implementation, proof and limits](evidence/gpt-oss/fused-prefill/README.md).

### GPT-OSS stable expert grouping

Grouped prefill now sorts routes once per layer, preserving exact row/route
order while reducing measured host synchronization calls from 769 to 73.
Full-model logits are bit-exact through 4000-token prompts. Four alternating
HTTP rounds show about 7–10% lower short-prompt latency, 2% lower 4K latency and
11% lower latency for the measured concurrent pair, with identical responses.
See [routing proof and matched measurements](evidence/gpt-oss/expert-routing/README.md).


## Standalone GGUF checkpoint import

Dense Qwen2/Qwen3 now load standalone GGUF files into damendaemon's native decoder.
Qwen2.5 Q4_0/Q8_0 and Qwen3-0.6B Q8_0 import/execution equivalence is qualified
on A2, including actual HTTP model swapping and the embedded tokenizer/chat
path. Quantized matrices retain their supplied values and scales. This is
format coverage; it does not remove existing architecture/precision limits or
qualify all GGUF formats. See [scope, exact fixtures and measurements](evidence/gguf/README.md).


Llama 3 BPE and Llama-style SentencePiece BPE now also load through the standalone
GGUF path. Mistral-7B-Instruct-v0.1's pinned original SentencePiece vocabulary
passes 512 native encoding and 1,024 decoding checks. A separate Q4_0 comparator
with those vocabulary scores preserves all 291 original packed weight tensors.
At 512 context, 160 independent HF complete-logit checks and 128 runtime-layout
checks pass. The optional CUDA C++ decode path passes another 128 layout checks
and 96 extended HF checks, worst RMS 0.4425%, with greedy agreement throughout.
The failed static-rank control for the older placeholder-score vocabulary is
retained. Native C++ SentencePiece now handles positional score ties without
inventing replacement scores. The unchanged published TinyLlama 1.1B Chat
Q4_K_M passes 2,048 native encoding, 4,096 full decoding and 2,048 continuation
checks, 160 independent model checks and 128 runtime-layout checks at context
512. Its normal server passes lifecycle checks and matches llama.cpp output at
concurrency 1/2/4, while remaining 26–79% slower on those measured workloads.
Subsequent normal model/layout and HTTP lifecycle proof reaches 2048 context,
including cross-model swaps. The same-file llama.cpp comparison at that context
retains long-prompt and concurrent output differences; all 96 independent HF
checks on those exact prompts agree with damendaemon. The optional residual path lowers
single-request latency 2.7–8.1% versus original damendaemon in that separate workload.
This does not establish general SentencePiece, unused-token or grammar support
on the positional path, longer-window semantics or performance parity.

### GLM-5.3-Flash text metadata contract

`python -m scripts.audit_glm_flash_checkpoint SNAPSHOT --names-only` audits
the pinned `zai-org/GLM-5.3-Flash` revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`. The official index matches
74,001 main-text tensor names. Its 1,760 MTP and 347 visual tensors are
explicitly excluded and remain unvalidated. Omitting `--names-only` checks
local safetensor headers against the shape/storage contract; neither mode reads
weight values or qualifies model execution. Only the names-only mode has been
run against the official snapshot; header tests use synthetic fixtures.

The contract distinguishes floating KDA projections, NoPE sparse attention,
floating indexer/latent expansion, FP8 E4M3 projections with FP32 block scales,
and mHC residual parameters. Quantization exclusions and layer schedules must
agree with that contract. Config/index reads and inventory geometry are bounded.

Shapes were independently transcribed from the pinned config/index and
[Transformers v5.17.0 glm5_next](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/glm5_next/modeling_glm5_next.py).
Source/index fingerprints are recorded in `tests/fixtures/glm53_flash_contract.json`.
The checkpoint's MIT license is preserved in `docs/licenses/glm53-flash-MIT.txt`.
Flash text execution, vision, MTP, quality and hardware qualification remain open
in [#57](https://github.com/filthyrake/damendaemon/issues/57).

`glm_flash_hyper.FlashHyperConnection` supplies bounded CPU FP32 residual
collapse/restoration and the Flash unweighted final stream mean. It normalizes
the flattened state before projecting, shares the existing Sinkhorn coefficient
calculation, and restores with source-to-destination matrix orientation.
Parameters are borrowed and checked for mutation-induced nonfinite values;
workspace/output limits are checked before input scans or arithmetic. Independent
Python scalar controls cover one and twenty normalization iterations, plus
noncontiguous state, asymmetric restoration, malformed inputs and budget failures.
These are synthetic primitive checks, not complete Flash text execution.

`glm_flash_kda.recurrent_attention` adds the functional CPU FP32 KDA recurrence
with per-key-dimension log decay, normalized query/key vectors and delta updates.
It returns independently owned final state and leaves all inputs unchanged.
Temporary/output budgets are admitted before scans or allocations; nonfinite
inputs, overflow, invalid gate ranges and malformed geometry fail explicitly.
Independent scalar controls and exact full/chunk/single-token continuation checks
pass on synthetic fixtures. The owned layer integration is described below.

`glm_flash_linear.FlashLinearAttention` composes floating checkpoint-shaped
projections, depthwise causal SiLU convolution, the bounded forget gate, KDA
recurrence, sigmoid-gated RMS normalization and output projection. Its CPU FP32
state owns convolution tails, recurrent matrices and an explicit cursor/capacity.
Full prefill, continued chunks, decode and reordered/shrinking/duplicated batches
are checked against independently assembled synthetic controls. Calls publish
replacement state only after all output checks pass; injected failures preserve
the previous state for retry. Closing releases the owned tensors; reuse requires
a fresh state. Retained-cache limits exclude temporary overlap, which is covered
by separate workspace admission. No official Flash weights have been executed.

`glm_flash_indexer.FlashIndexer` projects normalized index keys and pooling gates,
learns channel-wise weighted complete-pool keys, and selects causal pools with
visible incomplete tails. Scoring uses ReLU per head before signed head weighting.
CPU FP32 synthetic scalar controls cover learned pooling and scores; boundary
fixtures cover left/right padding, all-padding rows, ties and cache causality.
Valid tokens must form a contiguous interval; internal holes are rejected.
Exact ties select earlier pools deterministically. Returned int64 indices reserve
`top_k` complete-pool slots and `pool_size-1` tail slots with `-1` padding; this
canonical placement preserves selected sets rather than reproducing the upstream
variable tail offset. Workspace and output are separately bounded before scans.
This is a stateless indexer primitive, not sparse-attention/model qualification.

`glm_flash_attention` exposes explicit NoPE projection/selected-latent attention
entry points. The shared GLM engine accepts zero rotary width, retains selected
KV tile expansion and online softmax, and now rejects overflow in latent RMS
normalization instead of allowing a finite zero result. Independent dense masked
controls cover NoPE output and tile boundaries; packed E4M3/FP32-scale projection
fixtures agree with independently decoded floating controls. Existing rotary
GLM tests remain part of regression coverage. Full-model execution remains unqualified; owned layer composition follows.

`glm_flash_sparse.FlashSparseAttention` joins NoPE projections, pooled indexing,
selected latent expansion and output projection with owned latent/key/gate
histories. It admits the full declared cache capacity but grows actual retained
history with committed tokens. Temporary replacements and indexer scratch are
covered by separate workspace admission; explicit packed adapters retain their
own tile-workspace allowance in addition to the layer budget. No expanded full
KV cache is retained. CPU controls cover dense and packed projections, full/chunk/
decode, pool-boundary continuation, reorder/shrink/duplicate, release and rollback
after late failure. Inputs are unpadded at this layer; standalone indexer padding
support does not imply padded model admission. Official weights are unexecuted.

`glm_flash_ffn` makes Flash's clipped activation policy explicit for both floating
and packed dense/routed/shared FFNs: cap gate projection above the limit, cap up
projection on both sides, then apply SiLU and multiply. Ordinary GLM retains its
unclipped default. Router correction affects expert choice only; normalized
coefficients multiply after down projection, and the shared expert is added
without a route coefficient. Independent controls exercise clamp boundaries and
packed-versus-decoded outputs. Packed expert activation policies must match the
requested Flash limit. These CPU fixtures do not qualify official model weights.

`glm_flash_block.FlashBlock` composes either attention kind, two mHC mappings,
RMS norms and clipped FFN into a complete multi-stream residual block. Independent
composed equations cover both attention kinds; chunk/decode/reorder fixtures
exercise state continuity. A failure before attention commits remains retryable.
A failure after attention advances poisons the block, whose cursor remains at
the last complete block call; release is still available. Residual workspace is
additional to each component's declared scratch. Full-model composition remains
separate. Clipped expert activation now rejects nonfinite projections before
clamping, preventing a positive-infinite packed projection from becoming an
apparently finite result.

`glm_flash_model.FlashModel` supplies prepared CPU mixed-model execution from
embedding through KDA/sparse residual blocks to the unweighted stream mean,
final RMS normalization and output head. Aggregate cache admission combines
constant-size KDA histories with capacity-dependent sparse latent/index caches.
Teacher-forced synthetic logits agree with an independently assembled mixed
stack; packed attention and dense/routed/shared experts agree with decoded
controls. Full/chunk/decode and batch reorder/shrink preserve state. Partial
layer/head/reorder failures poison the model; failed cleanup remains owned and
retryable. Checkpoint loading, tokenization, generation, official weight quality,
vision/MTP, serving and hardware qualification remain separate gates.

`glm_flash_reference_loader.load_reference_model` loads admitted local main-text
snapshots into the mixed reference stack. Config/index/header and component policy
checks precede payload reads. Aggregate weight admission includes retained packed
bytes/FP32 parameters plus the largest source conversion overlap; shard count and
header bytes are bounded separately. KDA, indexer and mHC tensors stay floating;
main FP8 attention/FFN matrices stay packed with FP32 scales. Visual and MTP values
are excluded. Tiny two-shard fixtures execute against independently decoded model
controls and cover exact admission, malformed headers/index, changed files and
descriptor cleanup. The default 256 MiB weight budget does not admit the official
checkpoint; no official Flash weight values have been read or executed.

`glm_flash_text.FlashText` and `glm_flash_generation.FlashReferenceGenerator`
provide ordinary text and owned greedy CPU generation. The pinned Flash tokenizer
is byte-identical to the pinned regular GLM tokenizer: 154,856 mapped IDs within
154,880 model rows. Six literal prompts and token-ID cases were checked with the
official Flash template in a sandbox and the actual tokenizer. Flash media
branches differ and remain unsupported; tools are also rejected. Shared ordinary
format/parser/lifecycle code is reused with explicit Flash model/text binding and
nested configuration validation. Unmapped padded logits cannot be sampled.

Run `python -m scripts.run_glm_flash_reference SNAPSHOT --prompt 'Hello'` for the
explicit CPU reference command. It admits context/text and byte budgets before
loading weights, emits an unqualified CPU scope record, and releases abandoned
or failed generation state with owned cleanup retry. Tiny real checkpoint CLI
tests use a controlled head for deterministic lifecycle checks; no output quality
or official-weight execution is implied. Hardware tests remain paused.

### GLM-5.3-Flash prepared vision reference

`glm_flash_vision` independently describes all 347 visual tensor names in the
pinned Flash index and executes prepared CPU FP32 vision parameters. The pinned
[Transformers v5.17.0 architecture](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/glm5_next/modeling_glm5_next.py)
uses biased temporal patch projection, full axial rotary attention with Q/K
RMSNorm, clipped SwiGLU blocks, spatial convolutional downsampling, then a
LayerNorm/GELU/clipped-SwiGLU merger. Frames attend independently and patch rows
use spatial merge-block order. Attribution/licenses remain in
`docs/licenses/transformers-Apache-2.0.txt` and `glm53-flash-MIT.txt`.

The reference validates exact prepared shapes and CPU FP32 storage, parameter
bytes, token counts, quadratic attention scratch and output limits before
execution. Overflow/nonfinite values fail rather than being hidden by clipping
or normalization. Fixture controls separately express Conv3d/Conv2d, explicit
head attention, scalar rotary coordinates and the merger, checking intermediate
and final results plus image/frame isolation. No official visual payloads were
loaded. This does not qualify trained image semantics, preprocessing, language
image insertion, video support, HTTP, GPU execution or performance; existing
text-only loader and serving guards remain unchanged.

`glm_flash_images` adds the separate pinned Flash still-image processor for CPU
HWC uint8 RGB tensors. Unlike Qwen, it uses 14-pixel patches, CLIP normalization,
aligned aspect-preserving resize with bottom/right black padding, and duplicated
temporal patches. The implementation validates the entire batch's patch,
workspace and retained output limits before resizing or allocating output. CPU
bicubic interpolation retains uint8 semantics before normalization. Resize cases
come from the pinned v5.17.0 processor; scalar patch checks cover normalization,
padding and merge-block order. Fixture images also execute through the prepared
Flash vision encoder. Encoded files, network image retrieval, video and language
feature insertion remain outside this gate.

The Flash reference language model also exposes checked `forward_embeddings`
and owned `prefill_embeddings` methods. They share the token path's cache
ownership, capacity checks, cursor commit and partial-failure cleanup.
`glm_flash_image_embeddings.image_embeddings` creates owned input rows by
replacing each row's image placeholders with exactly its supplied CPU FP32 vision
features in prompt order. Feature counts, widths, storage and finite values are
validated before allocating embeddings. Text-only rows use empty feature matrices.

CPU fixtures now exercise RGB -> patches -> vision -> image insertion -> mixed
language logits, chunked image prefixes and ordinary token decode. This explicit
prepared-feature API does not render media chat, establish image identity or
semantic correctness, load official visual weights, or change the ordinary text
adapter's media rejection.

`glm_flash_vision_loader.load_reference_vision` loads only visual payloads from
an indexed or single-file snapshot. Exact names, floating shapes/storage,
quantization exclusions and whole-index consistency are checked first. FP32
retention, largest source conversion overlap and parameter-validation scratch
are admitted before payload reads. Descriptor and metadata limits remain
explicit; mutation or read failures close all opened descriptors. Mixed
FP16/BF16/F32 fixture snapshots match independently prepared vision controls.
Text/MTP payloads remain unread, and official visual weights have not been run.

`glm_flash_image_text.FlashImageText.image_chat_tokens` is a separate explicit
still-image prompt adapter. User content accepts strict text parts and prepared
`{"type":"image"}` markers; image grids supply the number of repeated image
tokens (`height * width / 4`) in prompt order. The pinned template's system,
reasoning and assistant behavior is retained. Three official local
prompt/tokenizer cases match exactly. Atomic media tokens, image/grid agreement,
conversation/part/image bounds and total expanded token limits are enforced.
Literal media markers in text, video/audio and URL parts are rejected. The
inherited ordinary `chat_tokens` still rejects multipart content. This adapter
binds placeholder counts; callers remain responsible for supplying the matching
images/features, and no network/file retrieval or serving admission is added.

`FlashImageReferenceGenerator.generate_images(messages, rgb_images, ...)`
composes the explicit CPU path: bounded RGB patches, grid-expanded chat tokens,
vision features, checked language embeddings and ordinary autoregressive decode.
It validates geometry/tokenizer/configuration agreement and context before vision
execution or cache allocation. The common generation loop owns streaming,
stop/length handling, padded-vocabulary exclusion, abandoned-iterator release and
retryable failed cleanup. Unsupported media-output markers are excluded from
image-response sampling. Existing plain-text generation remains available.
Fixture tests cover real tiny language execution and separately controlled
response/lifecycle cases. This reference API adds no HTTP or accelerator
admission and makes no trained image-semantic or performance claim.

The runnable reference command is:

```sh
python -m scripts.run_glm_flash_image_reference SNAPSHOT \
  --messages inline-image-messages.json --context 4096 --max-new-tokens 128
```

The JSON file contains alternating Flash chat turns with user text parts and
inline `image_url` PNG/JPEG data URLs. The Flash adapter reuses the bounded shared
decoder (four images, 8 MiB encoded bytes, 16 Mi pixels), preserves typed Flash
markers and image identities, and never fetches network URLs. Snapshot processor
metadata, image layout/output/workspace and expanded prompt/context are checked
before model weights load. Separate text/vision weight caps default to 256 MiB
apiece; cache/workspace/output/image limits are explicit command options. These
small defaults intentionally do not admit the full official checkpoint.

The command owns generator cleanup and emits `cpu_reference`, `still_image`,
`official_model_qualified: false` metadata before response events. A complete tiny
indexed checkpoint/inline-PNG command fixture passes; this is runnable software
coverage, not trained-model semantic or GPU qualification.

### DeepSeek CPU reference numerical failure boundaries

The floating reference normalization and hyper-connection paths reject
nonfinite inputs/parameters and intermediate overflow before reciprocal square
root, sigmoid, softmax, mixing normalization or dtype conversion can conceal
failed arithmetic. Finite `1e20` normalization inputs previously overflowed the
FP32 variance and returned zeros; overflowing finite head projections could
become saturated finite sigmoid coefficients. These cases now fail explicitly.
Final vocabulary projection and generation logits are checked before sampling.
Existing finite equations, output dtypes and numerical tolerances are unchanged.

CPU regressions cover these reproduced cases, FP16 cast overflow, prefill
release, partially advanced decode poisoning and generation cleanup. This is a
bounded correction under #119/#58, not official-weight or hardware qualification
or a claim that every arithmetic operation has a finite-intermediate guard.

### Isolated structured-output overhead (2026-10-15)

The remaining #15 overhead gate is now recorded in
[CPU setup and paired CUDA runtime evidence](evidence/structured/overhead/README.md).
Real-tokenizer production Constraint setup is separated from model activation;
matched fixed-logit/token controls isolate mask/matcher cost without conflating
it with different generated text. On the measured shared host, median cold
setup is2.052402s, cached setup44.972us, changed-schema setup2.194562ms, and
observed paired added A2 runtime cost233.939us/token. These are scoped component
measurements, not full-model throughput or exclusive-host latency guarantees.

## Explicit prepared CUDA GLM path

Issues #222–#233 add bounded CUDA packed projection/expert arithmetic, routed
composition, GLM post-down weighting, full/shared latent attention and indexer
caches, residual blocks, model logits, whole-model placement, native generation
and serialized loopback HTTP. `python -m scripts.serve_cuda_reference` accepts
an existing GLM snapshot and requires an indexed CUDA device plus separate
payload/allocator/context/cache limits. Ordinary production dispatch remains
unchanged. Prepared parameter tensors must remain immutable.

Tiny shared-A2 fixtures pass independent decoded CPU controls through
prefill/decode/reorder and poisoning/cleanup; separate zero-head fixtures prove
chat/SSE and owned command SIGTERM. Device status and timing identify this as
`cuda_prepared_reference`, with official-model/performance qualification false.
These are capability/lifecycle gates, not an official checkpoint, practical
context, quality, throughput or concurrent-serving qualification. Parent #56
stays open. [Exact results, stopped attempts and limits](evidence/packed-cuda/README.md).

The explicit CUDA reference path also accepts prepared Flash mixed linear/sparse
models and local Flash snapshots through `scripts.serve_cuda_reference`.
Whole-model placement validates raw payload and all CPU component metadata before
copying, retaining clipped FFNs, stream mixing and owned cache semantics. Tiny
shared-A2 scalar/model/command gates are recorded in
[CUDA evidence](evidence/packed-cuda/README.md). The text-only serialized command
keeps explicit device and separate weight/allocator/cache/workspace/output caps;
this does not qualify official weights, vision, practical context or throughput.

## Explicit prepared CUDA DeepSeek path

Issues #242–#253 compose bounded FP32 CUDA residual streams, compressed pooling,
learned selection, sink attention, owned cached decode, packed FP4/FP8 experts,
DeepSeek pre-down route weighting, full models and unique-payload placement.
The existing local DeepSeek loader now feeds `scripts.serve_cuda_reference` via
explicit placement and native ByteLevel/DSML single-sequence generation. For a
bounded setup, choose an indexed device and separate weight/allocator/cache
limits; `--projection-workspace-mib 8 --workspace-mib 256` supplies the loader's
existing tile allowance and the model's conservative transient allowance.
Workspace is an admission ceiling, not a claim about physical peak allocation.

Tiny shared-A2 decoded/scalar controls and loaded zero-head HTTP/SSE/SIGTERM gates
are retained in [CUDA evidence](evidence/packed-cuda/README.md). CPU default and
batch entrypoints remain separate. Status/timing report `cuda_prepared_reference`
with official-model and performance qualification false. This does not qualify
an official checkpoint, practical long context, generated tool accuracy or
concurrent-serving performance; #58 remains open.

Flash CUDA still-image support is explicit through `--flash-images` (#254–#256).
Pinned CPU RGB preprocessing feeds the bounded CUDA math-attention vision
encoder, ordered same-device feature insertion and the owned Flash image/tool
runner. Joint language/vision raw-payload admission precedes CUDA copies. The
command requires valid image-token/processor metadata; image budgets require the
flag. Separate `--vision-weight-mib`, `--max-patches`, `--image-workspace-mib` and
`--image-output-mib` bound the CPU preparation path, while `--device-weight-mib`
covers the combined device payload. Tiny feature/logit/stream controls and loaded
inline-image HTTP/SSE/SIGTERM proof are in the evidence directory. Official vision
quality, practical resolution/context and concurrent performance remain open.

Native host HTTP reports `execution=host_topology`, `device=distributed` and
`scope=host_reference`. `local_device` identifies only the observed coordinator
placement; `peer_devices_verified=false` means the wire protocol has not attested
remote placement. This applies even to a CPU coordinator because explicit CUDA
host adapters can be peers. Source-byte verification and topology identity do
not prove peer hardware. Local single-device CPU/CUDA runtime labels are unchanged.

`scripts.run_cuda_host_topology --family {glm,flash,deepseek}` exposes explicit
native text/tool `worker`, `complete` and `serve` roles using the same TLS files,
range plan, peers, source identity and `--native-chat` controls as CPU commands.
Each process must choose `--device cuda:N`; `--allocator-mib` bounds its PyTorch
allocator separately from `--device-weight-mib`, `--transfer-mib`,
`--device-workspace-mib` and CPU loader/cache/output bounds. Device weights must
leave allocator headroom. Caps exclude driver allocations and do not isolate
compute or protect other workflows by themselves; use the shared experiment
guard for bounded qualification. CPU commands still select CPU builders.
Flash image mode uses bounded CPU vision and an explicit collapsed CUDA embedding
row transfer before CPU image-feature insertion. Status reports `vision_device=cpu`
separately from the coordinator language device. No CUDA vision claim follows.

`scripts.audit_cuda_host_storage FAMILY SNAPSHOT --plan ranges.json
--host-budget-mib ... --device-budget-mib ...` audits an existing ordered CUDA
range plan from bounded headers only. It includes unique prepared device payload,
DeepSeek frequency sharing, host conversion overlap, aggregate slot cache and
separate workspace/output/transfer allowances. `--input-kind flash_embeddings`
includes the whole admitted prompt's collapsed embedding transfer before chunked
dispatch. It never reads weight payloads, probes hardware, rewrites cuts or reduces
context/slots. Exit 0 means declared limits fit, 1 means they do not, and 2 means
invalid metadata/controls. This is not execution admission or a measured memory
peak; driver/allocator overhead, unmeasured temporaries, other workloads and
coordinator vision/full image feature storage still need separate admission.

## Qwen3.5-MoE checkpoint contract and host text FFN

Issue #14, Qwen modern target (Qwen3.6-35B-A3B). The checkpoint contract
(`hybrid_moe_checkpoint`, merged in `1ecb9243`) pins the 693-tensor main-text
inventory — fused stacked experts, dense shared expert with a scalar
`shared_expert_gate` combine gate, and the 2x-width `q_proj` gate packing —
against the real 26-shard snapshot; MTP and vision tensors are enumerated and
excluded. `hybrid_moe_ffn` is the host reference FFN for that contract: FP32
CPU, top-k fused-expert routing through the audited `select_experts`, plus the
gated shared expert. It matches an independent per-token control across
sigmoid/softmax scoring with and without top-k renormalization
(`test_hybrid_moe_ffn`, torch CI job).

The qwen3_5_moe_text config pins no routing keys of its own (no
`norm_topk_prob`, no scoring field), so the routing semantics are pinned to
the eager reference implementation instead (Transformers 5.20.2
`Qwen3_5MoeTopKRouter`): FP32 softmax scoring, top-k on the probabilities,
and unconditional renormalization of the selected weights — no config gate,
unlike earlier qwen3_moe whose `norm_topk_prob` defaulted off. The FFN's
qualified default encodes exactly this; sigmoid scoring remains an explicit
unqualified option.

The executor (`scripts/experiments/qwen35_moe_layer_executor.py`) reads one
layer's seven MoE FFN tensors from the real 26-shard snapshot by bounded
slice reads (BF16->FP32, ~3.5 GiB peak, no staging copy, read-only NAS
untouched) and compares our FFN, an independent per-token control and the
eager upstream `Qwen3_5MoeSparseMoeBlock` over the same real tensors: layer 3
with a fixed-seed synthetic probe matches the upstream reference **bitwise
(max_abs 0.0, rms_rel 0.0)** and the independent control to 3.6e-08
(accumulation order). Real-router behavior, per-tensor digests and the probe
are recorded in `qwen35-moe-executor-layer3.json` under
`evidence/architectures`. Claimed: implementation semantics on real FFN
weights for this layer/probe. Not claimed: full-model execution,
real activations, GPU, performance or quality.

The four-layer text-path executor (`scripts/experiments/qwen35_moe_text_executor.py`)
extends the FFN cell to the full per-layer structure on real weights:
embedding, three gated-deltanet linear-attention layers, one full-attention
layer (partial rotary 64/256, qk-norm, q_proj-packed output gate, GQA 16/2),
the MoE FFN per layer, and the final norm, run by our port of the qualified
dense-hybrid math against the eager upstream `Qwen3_5MoeDecoderLayer` stack
(Transformers 5.20.2) over the same tensors — including both upstream norm
conventions (zero-centered `(1.0 + w)` layernorms/qk-norms, plain-weight
gated norm). Fixed-seed synthetic probe [1, 32, 2048]: linear layers
max_abs 2.1e-07..7.2e-07 (our sequential recurrence vs the upstream chunked
kernel), full-attention layer 6.6e-07, final hidden 1.7e-05 / rms_rel
2.5e-04 — sequential-vs-chunked accumulation divergence, no semantic
mismatch signature. Evidence `qwen35-moe-text-executor-layers0-3.json`
(71 tensor digests). Claimed: our text-path implementation matches the
upstream eager reference on real checkpoint weights for these layers and
this probe at float tolerance. Not claimed: mixer bitwise equality,
real activations, the full 40-layer run, GPU, performance, quality.

The full 40-layer streamed run (per-layer slice reads, ~8 GiB peak on
labnode3) completes the text-path qualification on real weights: all 40
layers agree with the eager upstream stack — per-layer max_abs up to 7.6e-06
(deepest full-attention layer), final hidden max_abs 3.0e-05 / rms_rel
2.4e-04; two linear layers show near-zero-output rms_rel spikes with
unchanged max_abs (metric inflation, not divergence). Evidence
`qwen35-moe-text-executor-full40.json` (692 digests). Claimed: full text-path
match vs upstream eager on real weights at float tolerance. Not claimed:
mixer bitwise equality, real activations, generation quality, GPU,
performance.

The expert-residency quantization budget (cell 6) measures weights-only
expert-bank quantization on real tensors across depth (layers 0/3/9/20/39)
with the audited simulations: E4M3FN worst max_abs 1.4e-03 (30.5 GiB bank -
does not fit the 24-GiB A30), E2M1+E8M0 worst 5.3e-03 / typical ~2e-03
(15.9 GiB - the only single-GPU residency option), normalized-Hadamard
rotation a wash (4.9e-03 worst). Router/shared-expert/combine-gate stay FP32
per the residency plan. Evidence `qwen35-moe-expert-fp4-budget.json`.
Claimed: the per-layer FFN-output budget only. Not claimed: compounding,
logits/generation impact, quality acceptability, GPU execution.

Cell 7 places the full 40-layer E2M1 expert bank on the A30 under the
shared-CUDA guard (15.95 GiB allocated vs 15.94 arithmetic) and composes
real layers through the prepared CUDA MoE stack: GPU vs the pinned-
quantization CPU control agrees to 1.1e-08 max_abs (op-order only), and vs
the FP32 reference lands exactly on the cell-6 budget (1.877e-03 /
5.268e-03) - the residency chain adds nothing to the measured quantization
error. Packing is exact by assertion (decode(pack(w)) == simulate_fp4(w)).
Evidence `qwen35-moe-a30-e2m1-residency.json`. Claimed: residency +
composition fidelity; not claimed: throughput, full-model GPU execution,
quality.

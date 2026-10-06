# Standalone GGUF import

Initial checkpoint qualification: Qwen/Qwen2.5-0.5B-Instruct, pinned HF revision
`7ae557604adf67be50417f59c2c2f167def9a775`, Q8_0 comparator file on NVIDIA A2,
PyTorch 2.6.0+cu124. Import uses the optional MIT-licensed `gguf==0.19.0` reader
package with additional bounds checks. Inference stays in dllm's decoder and
GPU kernels. No llama.cpp inference dependency is introduced.

Fixture SHA-256: `6810aa35d0eae4a02dea4031bcb983842888739028ebfdce797e16bf5b869d18`.
Reference llama-server executable SHA-256:
`344fe55b9043e7cd362b199a24fcbb9eba7e1d9513e17083a18df4190002401d`.

## Proof

- [A2 runtime report](qwen2-q8-a2.json): all 171 runtime weight tensors and 97
  scale tensors exactly match dllm's existing Q8 preparation from the pinned
  HF weights. Packed values and scales are copied, not requantized.
- Complete logits match bit-for-bit for eight generation steps at 1, 17, 129,
  and 505 prompt tokens, both with and without CUDA graphs (64 comparisons).
- Actual HTTP registration, chat, streaming, concurrent requests, cancellation
  recovery, host restoration after a model swap, and complete unload pass.
- Four alternating warm HTTP rounds, identical eight-token input and 16-token
  output: GGUF median 90.95 ms, HF-prepared Q8 median 93.94 ms. This is a small
  regression check showing no observed import-path penalty, **not evidence of
  a kernel speedup or a comparison with another engine**. Model activation and
  graph warmup are excluded.
- [Independent tokenizer report](qwen2-tokenizer.json): all 15 cases match
  llama.cpp's `/tokenize` with `parse_special=true` and `add_special=false`,
  including multilingual text, Unicode combining marks, emoji, whitespace,
  code, JSON/tool tags and control characters.
- The combining-mark and accented-text cases intentionally differ from the HF
  tokenizer's NFC-normalized output. GGUF follows the embedded tokenizer's
  behavior; it does not invent a normalizer absent from GGUF metadata.
- 257 CPU regression tests pass. Synthetic import tests cover F16/F32, Q4_0,
  Q8_0, Qwen3 Q/K norms, packed embeddings, tied heads, file-cache invalidation,
  incompatible fused formats, truncated/overlapping tensors, oversized metadata
  arrays, and non-finite packed scales.

Reproduce using `scripts/verify_gguf_tokenizer.py` and
`scripts/verify_gguf_runtime.py`. The latter requires a value-identical comparator
exported with `scripts/make_q8_comparator.py`; it does not assert equivalence
between unrelated quantizations.

## Boundaries

The importer accepts dense Qwen2/Qwen3 and Llama 3 BPE checkpoints in
little-endian GGUF v2/v3 with F16/F32 or Q4_0/Q8_0 tensors. The Llama extension
below includes explicit FP32 rotary-factor vectors. Later sections describe
experimental K-quants and Llama-style SentencePiece BPE. IQ formats, split
checkpoints, other architectures, other scaled-RoPE metadata, vision/audio
adapters and arbitrary GGUF tokenizer families are not implemented. Q/K/V and gate/up fused components must each use compatible
formats; distinct projections may use different supported formats.

Floating decoder weights follow the existing FP16 weight policy, including
normalization vectors; explicit rotary factors retain FP32. GGUF import is not an all-FP32 execution mode. CUDA packed embeddings now retain their original storage and dequantize only
selected token rows, rounding products to FP16 before any residual widening.
The Metal fallback expands Q8 embeddings to FP16 in bounded row chunks. Tied
packed heads share both values and scales with their CUDA embeddings. The metadata cache holds at most two
plans keyed by file identity, never mmap weight views. File changes invalidate
plans, and final host storage must match the preflight byte estimate.

Q4 is CUDA-only. Apple Silicon GGUF runtime qualification remains separate work;
existing native kernel qualification alone is not an import proof.
Broader import work is tracked in GitHub issue #43.


## Q4 and Qwen3 extension

[Qwen2.5 Q4_0](qwen2-q4-a2.json) and [Qwen3-0.6B Q8_0](qwen3-q8-a2.json)
pass the same 64 complete-logit comparisons and actual HTTP lifecycle checks.
Qwen3 uses pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`;
all 227 weight tensors and 113 scale tensors match native HF preparation.
Its independent head dimension and per-head Q/K normalization are exercised.
This establishes import/execution equivalence, not new quantization-quality or
long-context numerical qualification. Existing architecture limitations apply.
[Qwen3 tokenizer comparison](qwen3-tokenizer.json) also matches all 15 llama.cpp
cases, including the same two intentional differences from HF NFC normalization.

Fixture SHA-256 values:

- Qwen2.5 Q4: `128a1b5977611868f7756bc917c6bf20239f1abcf33da712cbc1f5a6e29ee217`
- Qwen3 Q8: `a4b9cf44accf020dbf7ed3fd0d7c8f6110cbee9754547e400b6cc2b39792fb5b`

The initial Q4 short-request check showed 127.58 ms for GGUF versus 104.42 ms
for HF-prepared Q4. That observation is retained rather than discarded.
A [controlled follow-up](qwen2-q4-residency-a2.json) uses three warm requests
before each sample, ten alternating rounds, both one- and two-model GPU
residency, and both initial loading orders. All responses remain exact:

| GPU models | Initial loading order | GGUF median | HF Q4 median |
|---|---|---:|---:|
| 2 | GGUF, HF | 95.31 ms | 93.83 ms |
| 2 | HF, GGUF | 94.96 ms | 98.58 ms |
| 1 | GGUF, HF | 94.07 ms | 94.52 ms |
| 1 | HF, GGUF | 93.50 ms | 92.35 ms |

The original 22% gap did not reproduce across these controlled cases. This
supports similar steady warm performance; it does not establish the cause of
the first result, and no kernel optimization was made between the runs. The
follow-up retains activation timings and GPU clock/power telemetry separately.

## Faster metadata loading

Profiling identified millions of temporary NumPy memmap view objects while
parsing tokenizer metadata. Bounded reads now create plain read-only ndarray
views over the same file buffer. All file bounds checks remain in place.

[Four alternating CPU trials](metadata-reader-benchmark.json) on the local Mac,
with an empty dllm metadata cache and potentially warm filesystem pages, reduce
median metadata/tokenizer-plan time from 4.363 s to 1.685 s (2.59x). Complete
plans, tensor descriptors, and serialized tokenizers match exactly. This is
**CPU model-loading work**, not GPU throughput or complete activation latency.
Reproduce with `scripts/benchmark_gguf_metadata.py` and the reader from commit
`1e5d2c1`. All 257 CPU regressions still pass after the optimization.


## Published checkpoint and exact-file downloads

`dllm pull`, `run`, and `add` accept `--filename` for one exact GGUF file within
a Hugging Face repository. The repository revision is resolved before fetching
the file; other quantization variants and safetensors copies are not downloaded.
GGUF file magic is recognized after symlink resolution, so HF cache blobs do
not need a filename extension. 263 CPU regressions pass, including the cache
symlink case and exact-download selection/rejection checks.

The real CLI downloaded official `Qwen/Qwen2.5-0.5B-Instruct-GGUF` at revision
`9217f5db79a29953eb74d5343926648285ec7e67`, selecting only
`qwen2.5-0.5b-instruct-q8_0.gguf`. Its SHA-256 is
`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
[Download result](gguf-official-qwen2-download.json).

[Actual A2 management proof](gguf-official-qwen2-management-a2.json) passes
registration from the resolved extensionless cache blob, disk/host/GPU loading,
active-stream unload, CLI management, model swapping, invalid-command recovery,
and preservation of the downloaded file. This published file contains a packed
Q8 embedding, unlike the value-matched comparator exports above.

The stronger numerical test independently dequantizes its GGUF tensors with
`gguf.quants`, maps them into the HF Qwen implementation, and compares complete
logits at each step. [FP16-reference report](gguf-official-qwen2-q8-reference-a2.json)
and [fixed-token FP32 control](gguf-official-qwen2-q8-fp32-reference-a2.json)
preserve all 64 greedy choices, but **both full stress gates fail after EOS**.
The math prompt produces `4` then an end-of-turn token; forcing continuation
past that stop reveals the discrepancy. All compared steps up to and including
the first EOS, and the two raw long-prompt traces, remain within the existing
0.5 absolute / 1% relative RMS gates.

The full FP16 comparison reaches 0.914 absolute / 5.43% RMS error. On exactly
the same token inputs, the FP32 reference reduces this to 0.184 absolute / 1.11%
RMS; the remaining RMS error still fails the unchanged gate. This is not claimed
as complete numerical qualification, and no threshold was relaxed. Follow-up
precision experiments are tracked under the existing Q8 quality issue #10 and
GGUF issue #43. Reproduction: `scripts/verify_gguf_reference.py` (use
`--reference-dtype float32 --teacher-trace <FP16-report>` for the fixed-input
control). The validation script exits unsuccessfully when its full gate fails.


## Rejected mixed-precision experiment

The [component probe](qwen2-precision-probe-a2.json) and
[rotary controls](qwen2-rotary-probe-a2.json) tested FP32 residuals, nonlinear
intermediates and rotary operations while retaining FP16 projections/KV and
packed Q8 weights. Only the combined policy passed the initial four-case trace.
Its [actual graph qualification](gguf-official-qwen2-q8-mixed-reference-a2.json)
passed all 64 comparisons, with maximum 0.167 absolute / 0.941% RMS error.

However, the [additional six cases](gguf-official-qwen2-q8-mixed-extended-a2.json)
found another post-EOS failure at 1.298% RMS. On exactly those token inputs,
[unchanged standard arithmetic](gguf-official-qwen2-q8-standard-extended-a2.json)
passes all 96 comparisons, with maximum 0.148 absolute / 0.9944% RMS error.
Both policies preserve every measured greedy choice. The mixed policy repairs
one stress case but regresses another; it is not a general numerical repair.

[Six alternating warm HTTP rounds](gguf-official-qwen2-precision-http-a2.json),
identical responses and 16 output tokens:

| Input tokens | Concurrent requests | Standard | Experimental mixed |
|---:|---:|---:|---:|
| 16 | 1 | 94.74 ms | 95.65 ms |
| 129 | 1 | 101.37 ms | 103.57 ms |
| 400 | 1 | 111.94 ms | 113.37 ms |
| 129 | 2 | 161.14 ms | 162.06 ms |

The 0.6–2.2% latency cost and the additional-case regression do not justify
adding this policy. **The candidate was removed from production; there is no
`DLLM_QWEN2_PRECISION` runtime option.** The [experimental patch](experimental-qwen2-mixed.patch)
is retained for reproduction against commit `c935d6e` in an isolated checkout.
`scripts/benchmark_qwen2_precision.py` requires that patch and fails explicitly
without it. `scripts/probe_gguf_precision.py` is a diagnostic using eager
executor-field changes, not a production arithmetic policy.

## Published tool-template discrepancy

The official Qwen2 GGUF's embedded template includes doubled outer braces in
its function-call JSON example. The pinned HF template uses single braces.
Independent rendering confirms the two-token difference. On A2 the published
GGUF followed the malformed example and produced:

```text
<tool_call>
{{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}
</tool_call>
```

The existing parser correctly keeps this malformed call as text. This is not
claimed as a successful tool round trip, and the importer does not silently
rewrite the embedded template. Explicit per-model template overrides are
tracked separately so a caller can select a canonical template. Tool reference
checks now read the actual GGUF template with the upstream reader and render
it independently through Transformers; they do not assume a different HF
snapshot has identical template text.

## Explicit chat-template overrides

`gguf-template-tools-a2.json` qualifies the Qwen3 Q8 comparator's embedded
chat template and the official Qwen2.5 Q8 GGUF with an explicit override from
HF tokenizer revision `7ae557604adf67be50417f59c2c2f167def9a775`.
The original official GGUF template stays unchanged on disk and remains the
loader default. Its doubled-brace tool example is not silently repaired.

Actual A2 HTTP registration, function arguments, independent tokenizer IDs,
tool-result history, streaming equality, four-way concurrency with distinct IDs,
model swaps, cancellation recovery, and full host/GPU eviction plus disk reload
passed. The maximum observed batch was four. No external tool was executed.
The report predates the proof script's additional malformed/conflicting HTTP
registration checks; those are covered by CPU/runtime/HTTP regression tests.
The complete local suite passed 268 tests after the override changes.

## CUDA packed embedding lookup

The loader now retains Q4_0/Q8_0 embedding values and scales on CUDA. A Triton
lookup reads only selected token rows and rounds each dequantized product to
FP16 before widening, preserving the previous expanded-embedding arithmetic.
Tied output heads share packed values and scales. Large embedding scales use a
bounded exact finiteness check; normal scales use a conservative range bound.
Metal's Q8 fallback keeps its existing FP16 expansion.

A2 proof:

- `packed-embedding-qwen2-controlled-a2.json`: official Qwen2.5 Q8 checkpoint,
  SHA-256 recorded in the report. Host weights fall from 797,246,208 to
  669,619,968 bytes (127,626,240 bytes / 121.7 MiB saved). GPU allocations for the
  measured four-slot graph executor fall by 127,806,464 bytes / 121.9 MiB.
- `packed-embedding-q4-a2.json` and `packed-embedding-qwen3-a2.json`: local
  Qwen2.5 Q4 and Qwen3 Q8 exports with quantized embeddings. Source HF revisions
  are the same pinned snapshots used above. These exports are generated with
  `make_q8_comparator.py --quantize-embedding`; they are not byte-matched to the
  ordinary HF path, which retains unquantized embeddings.
- Each report includes 60 bitwise kernel cases, changed-ID graph replay,
  128 bitwise full-model logit comparisons at prompt lengths 1/17/129/505 across
  fixed/paged KV and graph on/off, plus four-request decode and chunked prefill.
  The full-model comparator expands only the same stored embedding blocks;
  weights, quantization, context, tokens and arithmetic otherwise match.

Four alternating loading rounds, with the first round excluded from loading
medians, produced the following controlled Qwen2.5 Q8 measurements. Metadata,
filesystem and compiler caches are warm; these are not cold-machine startup
measurements. Activation includes host loading and a ready four-slot CUDA graph
executor. Warm generation is direct execution with 128 input / 16 output tokens,
not HTTP or post-first-token-only throughput.

| Median | Previous expanded embeddings | Packed embeddings |
|---|---:|---:|
| Host load | 1.665 s | 1.249 s |
| Activation | 2.839 s | 2.393 s |
| Warm generation | 83.99 ms | 84.15 ms |

Host load improved about 25%, activation about 16%; warm generation was
essentially unchanged. `packed-embedding-initial-a2.json` preserves the initial
run, whose loading measurements overlapped CPU fixture preparation and are
superseded by the controlled report. The local CPU/HTTP suite passes 270 tests.

`packed-embedding-tools-a2.json` additionally passes actual HTTP tool calls and
history, SSE equality, four-way concurrent unique IDs, cancellation recovery,
host restoration and disk reload for both packed Qwen3 Q8 and official Qwen2.5
Q8. Malformed and conflicting template registrations return HTTP 400 without
changing registry/residency state. This also completes the added negative HTTP
checks described in the template section above.

## Llama 3 standalone import

The loader now accepts Llama 3 BPE tokenization (`llama-bpe`, `llama3`,
`llama-v3`) and dense `llama` tensors. It restores the Q/K row ordering for both
floating and packed weights/scales, applies the embedded automatic BOS policy,
and preserves explicit `rope_freqs.weight` divisors in FP32. Unknown tensors,
noncausal attention, invalid rotary divisors and unsupported metadata still fail
explicitly. This does not add SentencePiece or Llama tool-call protocols.

Source: `unsloth/Llama-3.2-1B-Instruct` revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`, converted by the unchanged reference
converter at llama.cpp `836d57176dc699a726c55418e4f96b8ca628e1bf`.
Fixture hashes are in `llama32-gguf-sha256.txt`. The implementation follows the
format documented in that MIT-licensed project's
[Q/K permutation and rotary-factor export](https://github.com/ggml-org/llama.cpp/blob/836d57176dc699a726c55418e4f96b8ca628e1bf/conversion/llama.py)
and [BPE tokenization](https://github.com/ggml-org/llama.cpp/blob/836d57176dc699a726c55418e4f96b8ca628e1bf/src/llama-vocab.cpp).
No reference inference runtime is imported.

`llama32-gguf-tokenizer.json` matches both llama.cpp and the pinned HF tokenizer
on all 38 checks, including multilingual/Unicode text, digits, whitespace,
controls, code, special tokens, and automatic-special-token on/off.

A2 runtime qualification uses 512 context and four prompt lengths
(1/17/129/505), eight full-logit comparisons each, graphs on/off. The GGUF file's
explicit FP32 rotary divisors are a rounded representation of the HF Llama3
schedule, so not all logits are bitwise equal. The existing absolute <0.5 and
relative-RMS <1% gates plus exact greedy decisions apply to that explicit-factor
case; Qwen's bitwise import gate is unchanged.

| Format | Exact weights/scales | Bitwise logit steps / 64 | Max absolute | Max relative RMS |
|---|---:|---:|---:|---:|
| F16 | 99 / 0 | 20 | 0.01953 | 0.401% |
| Q8_0 | 99 / 65 | 30 | 0.01758 | 0.160% |
| Q4_0 | 99 / 65 | 32 | 0.01953 | 0.260% |

`llama32-gguf-f16-a2.json`, `llama32-gguf-q8-a2.json` and
`llama32-gguf-q4-a2.json` also pass actual HTTP
registration, matching HF/GGUF chat, SSE, concurrency, cancellation recovery,
host restoration and unload. These are import regressions on the same checkpoint
and precision, not throughput comparisons with another engine. GPU qualification
here stops at 512 tokens; the checkpoint's full advertised context is not proven
by this import test. The local CPU/HTTP suite passes 273 tests.

## Native K-quant kernel qualification

`kquant-kernels-a2.json` records 24 format/layout cases and 192 matrix
comparisons on A2, torch 2.6.0+cu124 / Triton 3.2.0 / gguf 0.19.0. Q4_K, Q5_K,
Q6_K, legacy Q4_0/Q8_0 and floating segments are decoded from compact byte rows.
Mixed segments cross output-tile boundaries. Exact FP16/FP32 expansion and
FP16-rounded embedding lookup match independent `gguf.quants` decoding. Matrix
products at 1/4/17/129 rows, including bias/no-bias, stay below 0.046% relative
RMS versus independently expanded PyTorch products (gate <0.1%, TF32 disabled).
Changed-ID embedding graphs and changed-input matrix graphs also pass.

This initial kernel report is not a speed claim.
`scripts/verify_kquant_kernels.py` reproduces it. The initial dynamically
branched segment implementation triggered a Triton 3.2 compiler failure;
static segments with masked native loads passed. No whole-model expansion or
requantization is needed by the kernels. FP32 multi-row matrix fallback expands
one projection temporarily; the initial FP16 implementation used native packed
Tensor Core kernels. The optimized dispatch below supersedes that choice.

## Llama Q4_K_M integration: correctness and initial performance gap

The same pinned Llama 3.2 1B F16 fixture was quantized using the unchanged
llama.cpp reference `llama-quantize` with `Q4_K_M`. SHA256:
`0f94fa98d163d9e70e97026f256da3876a9049e0d88eb6b8e880d754a01d7377`.
Its 96 Q4_K and 17 Q6_K tensors include mixed Q/K/V projections and a tied
Q6_K embedding/output head. Original coefficient bytes remain packed; there
is no requantization or inference-engine dependency.

`llama32-q4-k-m-reference-a2.json` and the extended report independently decode
those bytes through `gguf.quants` into a Hugging Face eager FP16 reference.
Across 160 full-logit comparisons with graphs on/off, raw prompts through 505
tokens, multilingual chat, arithmetic, code and conversation history, all greedy
decisions agree. Maximum absolute error is 0.1504 and relative RMS is 0.797%
(existing gates: <0.5 and <1%). This qualifies the Llama mixed-precision path at
512 context, not arbitrary models, Q5_K_M checkpoints or the advertised 131K context.

`llama32-q4-k-m-runtime-initial-a2.json` adds 36 comparisons against the same
coefficients expanded to FP16, covering four-way unequal-length batching,
fixed/paged KV, graphs on/off and chunked prefill. All greedy decisions agree;
maximum relative RMS is 0.222%. Real HTTP registration, chat, SSE, concurrency,
cancellation recovery, GPU unload/host restore and full disk reload pass.
The CPU/HTTP suite passes 277 tests.

The initial implementation is **not performance-ready**. Controlled alternating
HTTP measurements use the same coefficient values, A2, 512 context, two resident
models, three warmups and six measured pairs per case, with exactly 16 output
tokens and matching text/usage. These compare dllm's two execution paths, not
llama.cpp or vLLM:

| Input tokens / concurrency | Packed median | Expanded FP16 median |
|---|---:|---:|
| 16 / 1 | 5.498 s | 0.261 s |
| 129 / 1 | 7.992 s | 0.284 s |
| 400 / 1 | 12.983 s | 0.321 s |
| 129 / 2 | 11.177 s | 0.363 s |

Packed host weights use 799,727,744 bytes versus 2,471,628,928 bytes expanded.
The report's `passed` field denotes numerical/functional checks, **not a
performance gate**. Issue #45 remains open for kernel optimization and further
qualification. Preserve this initial report when adding subsequent measurements.

## K-quant decode and prompt-processing optimization

The initial per-projection measurements are retained in
`kquant-profile-initial-a2.jsonl`. Grouped header decoding, matrix-tile and
decode block-size experiments are in `kquant-profile-grouped-a2.jsonl` and
`kquant-profile-decode-a2.jsonl`. The format-specialization hypothesis alone did
not explain the old tile's latency; its timing was essentially unchanged.
Reducing repeated header work and changing execution shape produced the gains.
Larger packed matrix tiles and wider decode blocks were not uniformly better.

Default dispatch now uses packed vector products for one through four rows,
with bounded split reductions for widths >=4096. Larger batches expand only
the current projection for the matrix library. Persistent weights remain packed,
and GPU admission additionally reserves the largest temporary projection.
The original slow packed matrix kernel remains available to the profiling
harness, not selected by production dispatch.

The optional `native/kquant_gemv.cu` library uses warp-owned output rows,
paired coefficients/inputs and safe halfword loads over original GGUF bytes.
It performs explicit FP16 coefficient rounding followed by FP32 accumulation.
It does not repack or requantize weights, call another inference engine, or
change CUDA streams. The CMake `dllm-kquant` target was built on A2 with
`CMAKE_CUDA_ARCHITECTURES=86`, Release, and `--fmad=false`.
Set `DLLM_KQUANT_CUDA_LIBRARY` before starting the process to opt in; unsupported
precision/layout/batch shapes retain the default path. The ABI and buffer
layout are validated before launch. This is not an automatic runtime build.

`kquant-dispatch-kernels-a2.json` and `kquant-cuda-kernels-a2.json` each pass
384 matrix comparisons across 32 layouts, widths 256/768/2048/8192, FP16/FP32,
batch sizes 1/2/3/4/17/129 and bias/no-bias. Maximum relative RMS is 0.0204%,
below the unchanged 0.1% gate. Exact independent expansion/embedding and graph
replay with changed inputs pass. `kquant-cuda-coefficients-a2.json` additionally
compares 268,800 one-hot-probed coefficients bitwise against `gguf.quants`,
including Q4_K/Q5_K/Q6_K and a mixed group starting on a two-byte boundary.
Its library hash identifies the CMake build used for that proof.

`kquant-profile-cuda-scalar-a2.jsonl` retains the first C++ candidate;
`kquant-profile-cuda-vector-a2.jsonl` records the paired implementation.
The latter single-token matrix timings are below, using the same real Llama
Q4_K_M coefficients and CUDA-event timing over ten graph replays:

| Projection | Packed CUDA | Resident expanded FP16 |
|---|---:|---:|
| Q/K/V | 0.074 ms | 0.090 ms |
| Attention output | 0.051 ms | 0.063 ms |
| Gate/up | 0.314 ms | 0.414 ms |
| Down | 0.165 ms | 0.208 ms |
| Output head | 2.431 ms | 3.137 ms |

Four-row packed CUDA products still trail resident FP16 in this sweep.
These are isolated matrix measurements, not complete request performance or
comparisons against another engine. No whole-model Q5_K_M or broader-hardware
qualification follows from these kernel results.

`llama32-q4-k-m-runtime-triton-a2.json` repeats the full direct/HTTP proof on
the optimized default path. All numerical, greedy, batching, paging, chunking,
streaming and lifecycle checks pass; maximum relative RMS against the expanded
comparator is 0.119%. The same alternating 16-output-token HTTP protocol gives:

| Input tokens / concurrency | Optimized packed Triton | Expanded FP16 |
|---|---:|---:|
| 16 / 1 | 0.462 s | 0.269 s |
| 129 / 1 | 0.486 s | 0.285 s |
| 400 / 1 | 0.526 s | 0.324 s |
| 129 / 2 | 0.638 s | 0.373 s |

This removes most of the initial 5.5–13.0 second latency, but the default path
still trails resident FP16 by about 62–72%. Host weights remain 799,727,744 bytes;
the added temporary-projection reserve is 525,336,576 bytes. HTTP timings use
fixed KV; the top-level backend label records the last direct paged-KV test.
The optional C++ path's complete request measurements follow.

`llama32-q4-k-m-runtime-cuda-a2.json` passes the same full-model/HTTP/lifecycle
checks with the CMake-built native library selected. Maximum relative RMS against
the expanded comparator is 0.108%, with all greedy decisions equal. Its matched
alternating HTTP medians are:

| Input tokens / concurrency | Packed CUDA C++ | Expanded FP16 |
|---|---:|---:|
| 16 / 1 | 0.237 s | 0.262 s |
| 129 / 1 | 0.262 s | 0.285 s |
| 400 / 1 | 0.300 s | 0.324 s |
| 129 / 2 | 0.376 s | 0.366 s |

Single-request latency improves about 7–10% while retaining the original
compact weights; concurrency two is about 3% slower. This compares dllm paths,
not other engines, and excludes activation/cold-start time.

Fresh independent HF requalification for both optimized paths is recorded in
`llama32-q4-k-m-reference-{triton,cuda}{,-extended}-a2.json`: 160 full-logit
comparisons per path, no greedy mismatches, maximum relative RMS 0.824% and
maximum absolute error 0.137. The unchanged gates remain <1% and <0.5.
The initial and extended prompts, source checkpoint and 512-token scope are
unchanged. No broader architecture/context/hardware claim is implied.

## Same-file llama.cpp comparison

`llama32-q4-k-m-engines-a2.json` runs dllm with the optional CUDA library and
the unchanged pinned llama.cpp server sequentially on A2 in A/B/B/A order.
Each phase uses three warmup rounds and six measured rounds per case, identical
token IDs, exactly 16 output tokens, four slots, FP16 KV and 512 context per slot.
The comparator's `/props` confirms its actual slot limit; checkpoint, binary and
native-library hashes are recorded. These are matched stored-weight/KV formats,
not a claim that different engines use identical intermediate arithmetic.

llama.cpp prompt caching is disabled in every request; dllm prefix reuse and
speculation are disabled. `llama32-q4-k-m-engine-controls-a2.json` records the
reference device/offload lines and all 90 prompt-evaluation lengths across its
two phases: every request recomputed its complete 16/129/400-token input.
All generated text matches across engines, including the warmups.

| Input / concurrency | dllm latency | llama.cpp latency | dllm aggregate output tok/s | llama.cpp aggregate output tok/s |
|---|---:|---:|---:|---:|
| 16 / 1 | 0.242 s | 0.164 s | 65.99 | 97.79 |
| 129 / 1 | 0.263 s | 0.184 s | 60.72 | 87.02 |
| 400 / 1 | 0.298 s | 0.244 s | 53.68 | 65.62 |
| 129 / 2 | 0.379 s | 0.232 s | 84.51 | 138.09 |

Medians pool the twelve measured rounds per engine/case. Aggregate throughput
includes prompt processing and all requests in a round. dllm remains about
22–63% slower. At 400 input tokens, first visible output is approximately
0.100 s versus 0.102 s; the remaining gap is predominantly decoding. Shorter
prompts also expose dllm's temporary-expansion prefill cost. These measurements
do not establish parity or extrapolate to longer output/context or larger batches.

## Q5_K_M qualification gap

The same pinned F16 checkpoint was converted with the unchanged CPU reference
quantizer to Q5_K_M, SHA256
`e593d886d3cbdfed7a7c3c650866a7506d29f2e7d16218409316773f47c91db7`.
It contains 96 Q5_K and 17 Q6_K tensors. The initial native CUDA HF comparison
is retained in `llama32-q5-k-m-reference-cuda-initial-a2.json` and **fails** the
existing 1% relative-RMS gate: maximum 1.0343%, absolute error 0.08594, with all
64 greedy decisions equal. The failure occurs at a forced post-EOS step in the
short arithmetic trace; ordinary generation through EOS and the longer raw
traces stay below the gate. The threshold is unchanged and complete Q5_K_M
qualification is not claimed. Issue #45 tracks diagnosis and remaining proof.

Controlled teacher-token diagnostics are retained as
`llama32-q5-k-m-diagnostic-{triton,expanded,float32-reference}.json`.
The packed Triton path also fails the FP16 reference (1.0454% RMS), while the
same coefficients expanded into dllm's dense path pass (0.561%). Native packed
execution passes the higher-precision HF reference (0.608%) with the same forced
inputs. This narrows the issue to accumulation/precision behavior; the latter
result does not replace or erase the original failing gate.
`llama32-q5-k-m-reference-cuda-extended-a2.json` additionally passes all 96
extended FP16-reference comparisons, maximum RMS 0.700%, zero greedy mismatches.

The extended Q5 FP32-reference suite also passes all 96 comparisons (maximum
RMS 0.516%, no greedy mismatches), retained in
`llama32-q5-k-m-reference-float32-extended-a2.json`. The full executor/HTTP proof
in `llama32-q5-k-m-runtime-cuda-a2.json` passes fixed/paged caches, changed-input
graphs, batching, chunking, cancellation and model restore/reload. Its maximum
RMS against the same expanded coefficients is 0.125%. Packed host storage is
903,536,768 bytes. At 16/129/400 input tokens and one request, packed HTTP
latencies are 0.275/0.303/0.337 s versus expanded 0.263/0.288/0.324 s; at 129
tokens with two requests, 0.423 s versus 0.368 s. These additional passes do
not resolve the original forced post-EOS FP16-reference failure.

## Isolated residual-int8 experiment

`scripts/experiments/kquant_residual.cu` and its validation-only wrapper test
two dyadic int8 activation terms with integer dot products over original K-quant
bytes. This changes arithmetic: it is not the production FP16 coefficient path.
At this stage production did not select the candidate; the later qualified
opt-in is described below.

`kquant-residual-matrices-a2.jsonl` passes 80 independent matrix checks,
including changed-input graphs and wide-range inputs, at maximum RMS 0.0534%
against the unchanged 0.1% gate; zero-input graph replay also passes.
However, `kquant-profile-residual-a2.jsonl` shows the initial implementation
slower on all real projections than the validated CUDA implementation. Single-row
latency is approximately 7–45% worse, and four-row latency approximately 89–119%
worse. It is rejected for production selection. No full-model speed or quality
claim follows from these isolated kernel results.

The initial experiment (source at commit `ebcfe2e`) was built with:

```sh
nvcc -O3 -std=c++17 -arch=sm_86 --fmad=false -Xcompiler=-fPIC -shared \
  scripts/experiments/kquant_residual.cu -o results/libdllm_kquant_residual.so
```

The validation scripts select it only through
`DLLM_EXPERIMENTAL_RESIDUAL_LIBRARY`; ordinary dllm execution ignores that variable.
The later production option uses a separate variable and a narrower model scope.

The residual experiment's second revision removes repeated subgroup reductions
from the dot-product inner loop. `kquant-residual-v2-matrices-a2.jsonl` retains
the same independent matrix/graph gates. `kquant-profile-residual-v2-a2.jsonl`
shows single-row QKV/output/gate-up/head gains of approximately 3/8/14/25% over
production CUDA, but the 8192-wide down projection is 14% slower and all four-row
cases regress. Validation-only `--experimental-residual` therefore selects
single-row FP16 2048-wide K-quant projections and falls back elsewhere. Reports
record the experimental library hash and warn that ordinary backend labels do
not include this override. These early results precede the qualified opt-in described below.

Full-model residual-v2 reports are retained as
`llama32-q4-k-m-residual-v2-reference{,-extended}-a2.json`: all 160 greedy
comparisons agree with independent HF execution, maximum absolute error 0.140625
and RMS 0.894%, below the unchanged 0.5/1% gates. The runtime report
`llama32-q4-k-m-residual-v2-runtime-a2.json` passes batching, chunking, fixed/paged
caches, graphs, HTTP streaming/concurrency/cancellation and host/disk restoration.
Its batch-four decoding deliberately uses the production fallback; single-row
candidate decoding is covered by the HF suites. HTTP medians against identical
expanded coefficients are 0.211/0.236/0.275 s versus 0.259/0.284/0.321 s at
16/129/400 prompt tokens, and 0.366 versus 0.365 s for 129 tokens/concurrency two.
These are candidate-versus-expanded results, not a direct production-kernel
comparison. This report was obtained through the isolated validation dispatch.

The direct comparison is now retained in
`llama32-q4-k-m-residual-v2-http-abba-a2.json`. Production/candidate/candidate/
production server phases use identical GGUF bytes, prompt IDs, FP16 KV, context
512, four slots, 16 output tokens and disabled prefix/speculative paths. Each
phase has three warmups and six measured rounds per case. Both implementations
use the original packed weights; only the candidate's explicitly recorded
single-row arithmetic differs. All generated text matches across every phase.

| Input / concurrency | Production CUDA | Residual-v2 candidate | Median latency reduction |
|---|---:|---:|---:|
| 16 / 1 | 0.237830 s | 0.212148 s | 10.8% |
| 129 / 1 | 0.260549 s | 0.236988 s | 9.0% |
| 400 / 1 | 0.298261 s | 0.276075 s | 7.4% |
| 129 / 2 | 0.373792 s | 0.370382 s | 0.9% |

Medians pool twelve measured rounds per implementation/case. The concurrency-two
result is effectively a tie; batched decoding retains production dispatch. These
results do not establish another-GPU, another-model or llama.cpp parity claim.

`llama32-q4-k-m-residual-v2-executor-a2.json` adds explicit single-request
fixed/paged decoding with and without CUDA graphs, then reuses the slots for
batch-four and chunked prefill. All 68 comparisons pass: maximum single-row RMS
0.173%, batch-four RMS 0.107%, chunked RMS 0.0315% against the identical expanded
coefficients. This closes the candidate's single-request paged-decode coverage
missing from the original runtime harness; the production fallback remains
exercised by batch four.

The Q5 residual-v2 initial HF check is retained in
`llama32-q5-k-m-residual-v2-reference-a2.json` and still **fails**: maximum RMS
1.0288%, absolute 0.1015625, all 64 greedy decisions equal. It fails the same
forced post-EOS arithmetic trace at step six with graphs both off and on.
The candidate does not resolve the Q5 qualification gap. Its Q4 speed result
must not be presented as complete K-quant numerical qualification.

## Qualified optional residual backend

The shared CUDA implementation is now built as `dllm-kquant-residual`; explicit
`DLLM_KQUANT_RESIDUAL_LIBRARY` enables per-executor selection. Selection requires
CUDA, Llama `mixed` precision, hidden width 2048, context at most 512, and packed formats exactly
Q4_K/Q6_K. Any Q5-containing model is excluded. Only single-row 2048-wide
projections use it; all other projections retain ordinary dispatch. Ordinary
K-quant CUDA remains separately configurable for those fallbacks. Selection is
not a process-wide monkeypatch and does not affect expanded or other-family
models sharing the service. The validation entry points still permit broader
experimental arithmetic for diagnostics.

`llama32-q4-k-m-residual-production-runtime-a2.json` confirms the normal server
path selects `cuda-kquant-residual` and passes all 68 executor comparisons plus
HTTP/lifecycle checks. Single-request packed/expanded medians are approximately
0.211/0.261, 0.237/0.283 and 0.275/0.321 s at 16/129/400 input tokens;
concurrency two is 0.364/0.361 s. The separately recorded direct A/B/B/A comparison
above establishes the gain over ordinary packed CUDA. These are A2/512-context
Llama 3.2 1B results, not a universal backend performance claim.

The CMake-built library passes the same 80 independent matrix/changed-input graph
checks (`kquant-residual-production-matrices-a2.jsonl`). The normal production
selection passes the 96-case extended independent HF suite again, with the same
0.894% maximum RMS and exact greedy decisions
(`llama32-q4-k-m-residual-production-reference-extended-a2.json`). Reports record
the selected backend and rebuilt library hash. Local selection/isolation tests
exclude Q5, other families, FP32/standard precision and unsupported shapes;
the full CPU/HTTP regression suite has 281 passing tests.

## Qwen3-0.6B Q4_K_M

The existing pinned Qwen3-0.6B F16 conversion was quantized by the unchanged
reference CPU quantizer to `qwen3-small-import-q4_k_m.gguf`, SHA256
`9467d0bfdf4673a4182a6a02e8a66a89f1e2b5e214fcb120f1dc5068b83468df`.
It contains 169 Q4_K, 29 Q6_K and 113 F32 tensors. The source HF revision is
`c1899de289a04d12100db370d81485cdf75e47ca`.

`qwen3-q4-k-m-reference-cuda-{initial,extended}-a2.json` passes all 160 independent
FP16 HF comparisons: maximum RMS 0.841%, absolute 0.126953125, no greedy
mismatches. dllm uses its standard Qwen3 policy (FP16 matrices, FP32 residuals
and nonlinear operations). `qwen3-q4-k-m-runtime-cuda-a2.json` passes all 68
single/batched fixed/paged/graph/chunked comparisons and the full HTTP/lifecycle
suite, maximum RMS 0.203% against the identical expanded coefficients.
Packed host storage is 478,137,344 bytes. The residual library was configured
in this test but correctly remains unselected for Qwen3.

| Input / concurrency | Native packed CUDA | Same coefficients expanded to FP16 |
|---|---:|---:|
| 16 / 1 | 0.144926 s | 0.158667 s |
| 129 / 1 | 0.145993 s | 0.160588 s |
| 400 / 1 | 0.170793 s | 0.185060 s |
| 129 / 2 | 0.216639 s | 0.222944 s |

These are HTTP medians after warmup, 16 output tokens and 512 context, alternating
six measured rounds per implementation. Packed execution is about 8–9% faster
for single requests and 3% at concurrency two. This is a within-dllm comparison;
it does not establish parity with another engine or qualify long Qwen3 contexts.

### Same-file Qwen3 llama.cpp comparison and output divergence

`qwen3-q4-k-m-engines-a2.json` records sequential dllm/llama.cpp/llama.cpp/dllm
phases with the same file, prompt IDs, FP16 KV, four 512-token slots, 16 output
tokens and no prompt-cache reuse. The control log counts in
`qwen3-q4-k-m-engine-controls-a2.json` confirm all 90 reference requests
recomputed their full input. Pinned reference binary/hash and commands are in
the report. Medians pool twelve measured rounds per engine/case:

| Input / concurrency | dllm CUDA | llama.cpp | Generated text |
|---|---:|---:|---|
| 16 / 1 | 0.148477 s | 0.141501 s | Different wording |
| 129 / 1 | 0.152062 s | 0.154091 s | Identical |
| 400 / 1 | 0.173443 s | 0.195110 s | Identical |
| 129 / 2 | 0.216758 s | 0.203051 s | Identical |

The report deliberately retains `passed: false`: its exact-output assertion
fails on the shortest prompt. dllm continues with “a specific element” where
llama.cpp chooses “a value.” Each engine is stable across its own phases.
Stored weights/KV and workload match; intermediate arithmetic and the generated
sequences do not universally match. No clean cross-engine equivalence is claimed.

The diagnostic `qwen3-q4-k-m-engine-prompts-reference-a2.json` evaluates those
exact prompt IDs and 16 decode decisions with independent gguf.quants/HF eager
FP16 execution, graphs both off and on. All 96 greedy decisions match dllm,
maximum RMS 0.368%. Thus the differing sequence passes dllm's independent
reference gate; it is not evidence that dllm reproduced llama.cpp's arithmetic.
The 400-token matched-output case is approximately 11% faster in dllm, while
concurrency two is about 7% slower. The 129-token single case is roughly tied.

`qwen3-q4-k-m-reference-cuda-2048-a2.json` extends independent numerical coverage
to a 2041-token prompt followed by seven decode steps, with and without CUDA
graphs. All 64 comparisons pass, maximum RMS 0.662%, exact greedy agreement.
This is a 2048-context model-execution proof; the HTTP/paged lifecycle suite and
performance measurements above remain at 512 context.

`kquant-residual-batch23-profile-a2.jsonl` records a separate investigation of
batch sizes two and three. Batch three regresses on every measured projection.
Batch two shows potential gains in the sampled down projection and head but
regresses elsewhere. These microbenchmarks do not enable batched residual
production dispatch or establish whole-model gains.

The subsequent Llama residual 2048-context probe **fails** the unchanged RMS
gate on its 2041-token prompt: maximum RMS 1.876%, absolute 0.2890625, with all
64 greedy decisions still equal. The full failing result remains in
`llama32-q4-k-m-residual-production-reference-2048-a2.json`. Production residual
selection is consequently restricted to configured context capacities at most
512; larger capacities use ordinary dispatch. A same-teacher-trace ordinary
CUDA control is being evaluated to distinguish a residual-kernel regression
from the existing long-context precision behavior. This result does not erase
the passing 512-context proofs or establish a successful long-context fallback.

The ordinary CUDA control
`llama32-q4-k-m-native-reference-2048-a2.json` also fails on the same teacher
trace: maximum RMS 1.8766%, with exact greedy decisions. Its metadata confirms
that the residual option is unselected at 2048 context. The near-identical
result localizes the issue beyond the residual kernel itself; precision controls
are retained separately as they complete.

Repeated batch-two profiling across layers 0 and 2 is retained in
`kquant-residual-batch2-repeat-profile-a2.jsonl`. Three measurements per projection
confirm approximately 20% lower Q6 down-projection latency and 11% lower Q6 head
latency, while the Q4 down projection regresses. A validation-only candidate
therefore adds only two-row, pure-Q6 `(width, outputs)` shapes `(8192, 2048)` and
`(2048, 128256)` to the already-tested single-row policy. It is selected only by
`DLLM_EXPERIMENTAL_RESIDUAL_BATCH2=1` with the explicit validation entry point.

`llama32-q4-k-m-residual-batch2-runtime-a2.json` passes all 100 single/two/four-row
fixed/paged/graph/chunk comparisons (maximum RMS 0.173%) and HTTP/lifecycle checks.
Its concurrency-two packed/expanded medians are 0.361/0.366 s. This does not yet
establish an incremental gain over the production single-row residual option;
batched residual production dispatch remains disabled.

The direct incremental comparison
`llama32-q4-k-m-batch2-incremental-http-a2.json` compares production single-row
residual dispatch against the additional isolated batch-two shapes. All text
matches. Concurrency-two medians are 0.367534 versus 0.363293 s, approximately
1.15% lower latency. Unchanged single-request cases vary by roughly -0.6% to
+0.8%; this small incremental result is insufficient to promote another dispatch
policy. The candidate stays validation-only.

Long-context controls are retained in
`llama32-q4-k-m-context2048-control-{float32-reference,standard,expanded}.json`.
The mixed path versus higher-precision HF reaches 2.2715% RMS; standard precision
also fails; expanding the exact coefficients reproduces the ordinary packed
result at 1.8766% RMS. All greedy decisions remain equal. The independent rotary
probe (`llama32-gguf-rotary-control.json`) finds maximum frequency error
1.16e-10 and maximum phase error 2.38e-7 radians at position 2047. These controls
point beyond packed decoding and the imported rotary-factor vector; they do not
establish a passing long-context arithmetic policy.

The validation script now has an explicit `--weight-rounding kquant-float32`
policy for investigating the existing FP32 executor. Independent dequantization
retains unrounded FP32 K-quant projection coefficients while embeddings and
unquantized vectors preserve the loader's FP16 storage values. A tied packed
embedding/head therefore needs separate reference tensors under this policy.
The flag requires an FP32 HF reference and rejects uncovered legacy Q4_0/Q8_0
combinations. The prior `float16` reference policy remains the default; no
historical failed result or threshold is changed. Unit tests cover tied storage,
independent coefficients, vectors and rejected format combinations.

## FP32 K-quant numerical path

`llama32-q4-k-m-float32-reference-2048-a2.json` passes the same 2048-context
teacher trace under the existing `DLLM_LLAMA_PRECISION=float32` policy, using
explicit `--reference-dtype float32 --weight-rounding kquant-float32` independent
HF execution. All 64 greedy decisions agree; maximum absolute error is
0.00012112 and RMS 0.00160%. The old mixed/standard failures remain unchanged.
This establishes an accurate model-execution mode for this trace, not an HTTP
performance or full long-context lifecycle claim.

A separate validation-only FP32 residual-int8 library now preserves FP32 inputs
and outputs instead of rounding activations/results to FP16. ABI 101 deliberately
prevents accidental selection through the production FP16 ABI. Its independent
80-case matrix/changed-input/zero-graph proof passes at maximum RMS 0.00688%,
against the unchanged 0.1% matrix gate, in
`kquant-residual-f32-matrices-a2.jsonl`. This experiment is not a production option
and does not yet establish whole-model accuracy or speed.

The FP32 candidate subsequently passes the independent 2048-context teacher
trace in `llama32-q4-k-m-residual-f32-reference-2048-a2.json`: maximum RMS 0.281%,
absolute 0.03632, all 64 greedy decisions equal. The unchanged production FP32
path remains more accurate on this trace; the candidate remains below the
original 1% / 0.5 gates. `kquant-residual-f32-matrices-range-a2.jsonl` additionally
checks activation reconstruction from subnormal magnitudes through 2^100;
errors stay within 0.01% of each group's maximum plus one minimum subnormal.

`kquant-residual-f32-profile-a2.jsonl` compares real projections in layers 0 and 2
and the output head, batches 1/2/4, using identical unrounded FP32 coefficients.
The candidate is 1.5–3.9x faster than ordinary packed FP32 projection dispatch
across these cases. Single-row head time falls from 5.283 to 1.831 ms, and
four-row head time from 8.249 to 4.423 ms. These are matrix timings, not a
whole-model speed claim; full batching/lifecycle and HTTP comparisons remain
necessary before production integration.

The extended Q4 2048-context suite now passes 96 additional independent HF
comparisons, with exact greedy agreement and maximum RMS 0.142%
(`llama32-q4-k-m-residual-f32-reference-2048-extended-a2.json`). Q5's initial
64-check FP32 suite also passes, maximum RMS 0.128%; its historical mixed/FP16
failure remains a separate open qualification boundary.

`verify_kquant_fp32_replay.py` records ordinary packed FP32 outputs in a separate
process, then replays identical teacher tokens through the candidate. Q4 passes
100 fixed/paged, graph/eager, one/two/four-row and chunked-prefill comparisons at
512 context, maximum RMS 0.0362%, with all greedy decisions equal. The archive
hash and recipe are retained in `llama32-q4-k-m-fp32-packed-baseline-a2.json` and
`llama32-q4-k-m-residual-f32-replay-a2.json`. This isolates runtime behavior;
independent model accuracy comes from the HF reports above.

The actual server comparison `llama32-q4-k-m-residual-f32-http-a2.json` uses
ordinary packed FP32 versus candidate FP32, A/B/B/A isolated processes, identical
checkpoint/prompt IDs, FP32 KV, 512 context, 16 output tokens, three warmups and
six measured rounds per phase/case. IEEE FP32 library settings are explicit.
All generated text matches. Median request latency:

| Input tokens / concurrency | Ordinary FP32 | Candidate FP32 | Speedup |
| --- | ---: | ---: | ---: |
| 16 / 1 | 0.483025 s | 0.247496 s | 1.95x |
| 129 / 1 | 0.596491 s | 0.362491 s | 1.65x |
| 400 / 1 | 0.801102 s | 0.568457 s | 1.41x |
| 129 / 2 | 0.891643 s | 0.623102 s | 1.43x |

Every phase also passes host restore, alias swapping with one GPU-resident model,
disk reload, and early stream cancellation followed by recovery. This gain is
over dllm's ordinary FP32 path; it is not a same-precision win over llama.cpp or
the earlier FP16 residual option. FP32 prefill remains expensive. The candidate
is still validation-only while Q5 coverage and production integration proceed.


Q5's extended FP32 independent suite also passes all 96 comparisons at 2048
context (maximum RMS 0.1754%, absolute 0.01534). Its separate baseline/replay
reports pass all 100 runtime cases with maximum RMS 0.0564% and exact greedy
agreement. See `llama32-q5-k-m-residual-f32-reference-2048-extended-a2.json`,
`llama32-q5-k-m-fp32-packed-baseline-a2.json`, and
`llama32-q5-k-m-residual-f32-replay-a2.json`.


## Opt-in FP32 production dispatch

The FP32 arithmetic now lives in `native/kquant_residual_f32.cu`, built as
`dllm-kquant-residual-f32`. The experimental source includes that implementation.
`DLLM_KQUANT_RESIDUAL_F32_LIBRARY` selects a per-executor callback only for CUDA
Llama hidden width 2048, FP32 precision, context <=2048, and Q4/Q6 or Q5/Q6
layouts. Shapes outside rows 1–4 and widths 2048/8192 fall back. ABI 101 remains
separate from FP16 ABI 1; loaded modules stay alive for captured CUDA graphs.
No global production monkeypatch is used. Unit tests enforce precision, family,
context, format, shape, opt-in and per-executor library boundaries.

Both `llama32-q{4,5}-k-m-residual-f32-production-replay-a2.json` reports pass the
100-case runtime gate with the normal `cuda-kquant-residual-f32` backend label.
The Q5 production HTTP comparison passes all output/lifecycle checks using the
normal `dllm` CLI in every phase. Ordinary/candidate FP32 medians are:

| Input tokens / concurrency | Ordinary FP32 | Production residual FP32 |
| --- | ---: | ---: |
| 16 / 1 | 0.536503 s | 0.273409 s |
| 129 / 1 | 0.650599 s | 0.388951 s |
| 400 / 1 | 0.853740 s | 0.595127 s |
| 129 / 2 | 0.942085 s | 0.648724 s |

These represent 30–49% lower latency for the measured Q5 workloads. The raw
report is `llama32-q5-k-m-residual-f32-production-http-a2.json`; its library hash,
commands, exact prompts, measured rounds and lifecycle outcomes are retained.

Q4's normal-CLI confirmation also passes every output and lifecycle check in
`llama32-q4-k-m-residual-f32-production-http-a2.json`. Ordinary/production FP32
medians are 0.483638/0.248301 s (16/1), 0.598773/0.365971 s (129/1),
0.805575/0.572631 s (400/1), and 0.895685/0.628896 s (129/2). This reproduces
29–49% lower latency without the experimental wrapper. Both production replay
reports identify the FP32 residual backend; the server reports retain the exact
CMake-built library hash and the normal CLI command for each phase.

## FP32 prefill experiments

`scripts/experiments/kquant_prefill_f32.py` keeps three alternatives separate
from production. Every comparison retains unrounded FP32 projection coefficients
and FP32 results; it does not substitute the FP16 reference policy.

* `kquant-prefill-f32-profile-a2.jsonl`: 105 matrix cases compare IEEE FP32
  against three-term TF32 products, with transient expansion or repeated packed
  tile decode. All pass (max RMS 0.0000874%). Expanded TF32 improves 16-token
  cases but regresses at 400 tokens; direct packed tile decode is much slower.
  Neither is selected.
* `kquant-prefill-fp16x3-profile-a2.jsonl`: split both operands into scaled
  high/low FP16 terms and sum high*high, high*low and low*high products in FP32.
  The omitted low*low term makes this an approximation, not bitwise FP32.
  Longer cases improve, but preparing expanded weights hurts short prompts.
* `kquant-prefill-fp16x3-fused-profile-a2.jsonl`: decode packed weights directly
  into the high/low buffers, eliminating the temporary FP32 matrix. With a
  fixed 32-row tile below 256 tokens and 64-row tile above, the five tested
  real projections improve 23–39% at 129/400 input rows. The 16-row cases are
  mixed, so validation dispatch retains ordinary prefill below 128 rows.

The fused candidate with production FP32 residual decode passes all 64 checks
on the previously qualified 2048-context teacher trace, maximum RMS 0.1192%,
absolute 0.01518, exact greedy agreement. The independent HF report is
`llama32-q4-k-m-prefill-f32-reference-2048-a2.json`. These profile/reference
results alone do not establish server speed or paging/lifecycle qualification;
those checks are separate. Production prompt processing is unchanged.

The first extended runtime-baseline attempt retained completed batch requests
before its independent 384-token chunked request and correctly hit the 1024-token
page-pool limit. `llama32-q4-k-m-fp32-long-baseline-capacity-control-a2.json`
preserves that harness failure. The extended recipe now releases completed
requests before the chunk case; this does not increase the pool or weaken
admission. The original 512-context replay recipe remains unchanged.

After correcting the harness lifecycle, the long-prefill replay passes all 100
fixed/paged/graph/batch/chunk cases, maximum RMS 0.0362%, with identical greedy
decisions (`llama32-q4-k-m-fp32-long-baseline-a2.json` and
`llama32-q4-k-m-prefill-f32-replay-a2.json`). This recipe uses 128/256/384-token
prompts and 128-token chunks so it exercises the candidate prefill threshold.

The incremental server comparison `llama32-q4-k-m-prefill-f32-http-a2.json`
uses production FP32 residual decode in both engines, changing only prefill.
All text and lifecycle checks pass. Baseline/candidate medians are
0.248215/0.248818 s (16/1, unchanged path), 0.363922/0.321678 s (129/1),
0.568369/0.473856 s (400/1), and 0.624352/0.544167 s (129/2). The changed cases
improve 12–17% beyond the residual decode gain.

A subsequent targeted range check found that multiplying the accumulator by
one operand's scale before the other can overflow on finite 2^115 inputs even
when the final result is representable. The failing
`kquant-prefill-range-original-a2.json` is retained; that implementation is in
commit e65219f. Rescaling now combines the power-of-two exponents and uses
`ldexp` once. Production promotion requires range, model and server confirmation
of that corrected implementation; the earlier HTTP gain alone is insufficient.

The corrected range report `kquant-prefill-range-fixed-a2.json` passes every
case from 2^-140 through 2^115 against FP64 products of the same decoded FP32
weights. Repeated matrix profiling remains 22–38% faster at 129/400 rows
(`kquant-prefill-fp16x3-range-fixed-profile-a2.jsonl`). Corrected Q4 extended and
Q5 initial/extended independent HF suites all pass at 2048 context, with exact
greedy agreement; Q5's worst RMS is 0.1754%.

The corrected kernels now live in `dllm/kquant_prefill_kernels.py`. The
experimental entry point calls the same implementation. The normal executor's
separate per-model selector uses `DLLM_KQUANT_PREFILL_F32=1` only for CUDA Llama
hidden width 2048, FP32 precision, context <=2048, Q4/Q6 or Q5/Q6 layouts, and
rows >=128 at widths 2048/8192. Decode and dense-weight dispatch stay separate.
The normal production range, Q4/Q5 100-case long-prefill replay, and Q4
independent 2048-context model reports pass. Backend labels and kernel hashes
are retained in the `*-prefill-f32-production-*` reports. Normal-server timing
and lifecycle confirmation of this final dispatch are recorded separately when
complete. The local regression suite passes 289 tests.

Final normal-server confirmation passes for both Q4 and Q5 in
`llama32-q{4,5}-k-m-prefill-f32-production-http-a2.json`. Both phases use the
normal CLI; backend assertions confirm residual decode in both engines and
compensated prefill only in the candidate. Every generated text and lifecycle
check passes. These corrected-kernel A/B/B/A medians compare residual decode
alone against residual decode plus compensated prefill:

| Model | Input / concurrency | Decode only | Decode + prefill |
| --- | --- | ---: | ---: |
| Q4_K_M | 16 / 1 | 0.248831 s | 0.249036 s |
| Q4_K_M | 129 / 1 | 0.364477 s | 0.323720 s |
| Q4_K_M | 400 / 1 | 0.570949 s | 0.479223 s |
| Q4_K_M | 129 / 2 | 0.626377 s | 0.546656 s |
| Q5_K_M | 16 / 1 | 0.275045 s | 0.274001 s |
| Q5_K_M | 129 / 1 | 0.391113 s | 0.355199 s |
| Q5_K_M | 400 / 1 | 0.598608 s | 0.509351 s |
| Q5_K_M | 129 / 2 | 0.652573 s | 0.577975 s |

The changed longer-prompt cases improve 9–16%; the unchanged 16-token path is
approximately tied. This is an incremental FP32 gain, not a cross-engine or
cross-precision parity claim. Independent model coverage extends to 2048;
server, paging and lifecycle evidence remains at 512 context.


## Llama-style SentencePiece BPE

Issue #46 tracks full-model qualification. The runtime reconstructs the embedded
BPE vocabulary using the existing Rust `tokenizers` dependency; no SentencePiece,
protobuf, Transformers or llama.cpp runtime dependency is added. This supports
identity normalization, score-ranked BPE, byte fallback, BOS/EOS and control-token
boundaries. Unigram, custom normalization and tied non-whitespace merge scores
are rejected. The source's unknown token may be typed UNKNOWN or CONTROL.

The original Mistral-7B-Instruct-v0.1 `tokenizer.model` is pinned at
`ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca`, SHA-256
`dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`.
`scripts/prepare_sentencepiece_vocab.py` creates a vocabulary-only GGUF with no
weights; it is not an inference model. `scripts/reference/gguf_tokenizer.cpp`
links the unchanged llama.cpp API at revision
`836d57176dc699a726c55418e4f96b8ca628e1bf` and captures encoding and both decoding
modes. Reference library SHA-256:
`892b2900e612677241b33112f583d95bc89ac3b8da7b9094b1515b9ec93ce5c7`.

The initial 128 encodings and 256 decodings pass, including serialization/reload.
The expanded report `mistral-spm-tokenizer-expanded-reference.json` passes 512
encodings and 1,024 decodings, including whitespace runs of 1–64 spaces and
control boundaries. Requests and native oracle JSONL are retained alongside it.
A decoder adapter preserves the leading space when a leading control token is
hidden; the unwrapped decoder fails 325 expanded checks. The raw pinned HF
`tokenizer.json` control differs in 88 initial encoding cases; this is not a
claim about a fully configured HF AutoTokenizer.

The existing older Mistral Q4 comparator instead contains placeholder -1000
scores for every token. Ranked merges disagree with native positional tie
handling in 36/128 encodings (`mistral-spm-placeholder-score-control.json`).
That input is now rejected explicitly; this failed control predates the guard.
`scripts/prepare_gguf_vocab_comparator.py` creates a separate comparator with
the original pinned vocabulary metadata, requiring identical vocabulary IDs
and texts. `mistral-spm-q4-comparator.json` proves all 291 weight tensors retain
identical type, shape and bytes. The original file is unchanged. This does not
qualify arbitrary equal-score vocabularies, nor claim equivalence between the
two tokenizers. The comparator has no sliding-window metadata, so full-model
checks below 4096 cannot establish Mistral's longer-context window semantics.

Score-based merge reconstruction follows the approach in Apache-2.0
Transformers `convert_slow_tokenizer.generate_merges`; native behavior is checked
against MIT-licensed llama.cpp. Both are validation references rather than
inference engines used by dllm. The local regression suite passes 295 tests.

The initial full-model A2 report `mistral-spm-q4-reference-a2.json` passes all
64 complete-logit comparisons (four prompts, eight steps, eager and graph),
with exact greedy-token agreement, maximum absolute error 0.08594 and maximum
relative RMS 0.2114%. The independent reference dequantizes the same Q4_0 bytes
with `gguf.quants` and executes HF Mistral eager attention in FP16. dllm uses
its default Llama mixed-precision policy. Context is 512. The new
`--serial-reference` verifier mode saves CPU logits and releases the expanded
HF GPU weights before loading dllm, allowing this check on the A2 without
altering either runtime's arithmetic. This is numerical proof, not a timing
comparison or longer-window qualification.

The extended independent suite also passes all 96 checks with greedy agreement,
maximum absolute error 0.078125 and RMS 0.3928%
(`mistral-spm-q4-reference-extended-a2.json`). Inputs include multi-turn recall,
Unicode, code, a system message and repeated raw prefixes.

`scripts/verify_gguf_layout_runtime.py` checks batching (1/2/4), fixed/paged KV,
graph/eager execution and 128-token chunked prefill against separately captured
single-request fixed eager traces. It keeps one executor resident at a time.
Both the ordinary Q4 path and optional CUDA C++ Q4 decode pass all 128 layout
comparisons, with exact greedy agreement and maximum RMS 0.2654%:
`mistral-spm-q4-layout-a2.json` and `mistral-spm-q4-native-layout-a2.json`.
These are layout checks, not a substitute for the independent HF model oracle.
Native library SHA-256:
`29dffba1428aab9846a5b71464855a4d48e6fbd6b9ec8a89a47f64b1ef0dd063`.


The native Q4 path additionally passes all 96 extended independent HF checks,
maximum absolute error 0.089844 and RMS 0.4425%, with exact greedy agreement
(`mistral-spm-q4-native-reference-extended-a2.json`). The report identifies and
hashes the selected native library. The merge reconstruction attribution and
Apache-2.0 text are retained in `dllm/gguf_sentencepiece.py` and
`docs/licenses/transformers-Apache-2.0.txt`.

The first same-file normal-server A/B/B/A comparison is preserved in
`mistral-spm-q4-http-a2.json`. Both dllm phases pass cross-checkpoint
Mistral→Qwen3→Mistral switching, host restore, full disk reload, cancellation
recovery, streaming and concurrent chat. The whole comparison remains
`passed: false`: the 16-token prompt generates different words across engines;
the 400-token prompt differs by an initial space. Medians for 16 output tokens:

| Input / concurrency | dllm native Q4 | llama.cpp | Text equal |
| --- | ---: | ---: | --- |
| 16 / 1 | 0.720051 s | 0.494891 s | No |
| 129 / 1 | 1.029845 s | 0.648294 s | Yes |
| 400 / 1 | 1.245919 s | 0.958910 s | No, leading space |
| 129 / 2 | 1.524126 s | 0.803146 s | Yes |

Both engines use the same file, explicit token IDs, 512 tokens per slot, four
slots and FP16 KV. dllm uses its qualified mixed policy; this does not imply
identical internal arithmetic. The matching 129-token cases show dllm 59%/90%
slower at concurrency 1/2. No performance-parity claim is made.

The independent HF replay of those exact benchmark IDs passes all 96 checks,
with exact greedy agreement, maximum absolute error 0.023438 and RMS 0.0812%
(`mistral-spm-q4-engine-prompts-reference-a2.json`). The differing short-prompt
words therefore do not reproduce against the independent reference.

The leading-space case exposed a serving boundary error. Full-text native
`llama_detokenize` removes an artificial initial space; generated continuations
use `llama_token_to_piece` with no such removal. `decode_output` now selects a
continuation decoder for GGUF SentencePiece in both serialized and batched
runtime paths, leaving ordinary full-text decode intact. The updated native
adapter verifies 512 continuation outputs, 512 encodings and 1,024 full-text
decodings (`mistral-spm-generated-tokenizer-reference.json`, accompanying oracle,
and the earlier expanded request corpus). Scheduler streaming and fallback
regressions pass; the complete local suite has 298 tests. The earlier failed
HTTP report is intentionally retained; post-fix server comparison is separate.

Normal-server post-fix confirmation passes for all three benchmark prompts,
both streamed and nonstreamed, against independently qualified HF token traces
(`mistral-spm-q4-generated-http-a2.json`). The 129- and 400-token cases now also
match the retained llama.cpp text exactly. The 16-token wording difference
remains and is not hidden by this narrower passing gate. Initial activation
in this one warm-filesystem observation took 13.589 s; it is not a cold-disk or
multi-run loading benchmark.

## Legacy Q4 decode bottleneck and isolated integer-dot candidate

`mistral-spm-q4-profile-b{1,2}-a2.json` profiles normal graph execution. Native
Q4 GEMV consumes 40.81 of 41.94 ms of measured GPU work per single-request decode
step. At batch two, Q4 GEMM consumes 46.40 of 47.79 ms per step. These are profiler
observations, not standalone end-to-end throughput measurements. Field labels
were corrected from `per_token` to `per_decode_step` without changing values;
a batch-two step emits two tokens. Both profiles use the normal loader, with
about 9.0 s host preparation and 3.1–3.2 s executor creation observed individually.

`scripts/experiments/q4_residual.cu` is an isolated integer-dot experiment using
dllm's existing two-term dyadic int8 activation quantizer. It preserves packed
Q4_0 bytes and scales, but applies coefficient scaling in FP32 rather than
rounding every dequantized coefficient to FP16. This arithmetic difference is
explicit and requires independent full-model validation before promotion.

`q4-residual-matrices-a2.json` passes 108 matrix checks against FP16-rounded
independent coefficients, covering six shapes, rows 1/2/4, bias on/off and
activation exponents -14/0/10. Maximum RMS is 0.06186%. On the five large
single-row projections the candidate takes 17–24% less GPU time than production.
Batch-two/four cases do not justify selection; the batch-two down projection is
2.10x slower. All timings and rejected shapes are retained. The experimental
model hook therefore selects only one row, widths >=4096 divisible by 256, and
outputs >=1024. Normal production dispatch is unchanged.

Build the experiment separately with CUDA `nvcc -O3 -std=c++17 -arch=sm_86
-shared -Xcompiler -fPIC scripts/experiments/q4_residual.cu -o PATH.so`.
Set `DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY=PATH.so` only in an explicit validation
process. `scripts/profile_q4_residual.py` reproduces matrix checks and timing;
full-model verifiers use `--experimental-q4-residual`.

The first single-row integer-dot candidate passes all 96 extended independent
HF checks (maximum absolute 0.085938, RMS 0.4182%, exact greedy agreement) and all
128 runtime-layout checks (worst RMS 0.2653%). Reports are
`mistral-spm-q4-residual-reference-extended-a2.json` and
`mistral-spm-q4-residual-layout-a2.json`. End-to-end comparison is separate.

A diagnostic `DLLM_Q4_RESIDUAL_BYPASS_L1` compile-time variant keeps weight loads
out of L1; it is a distinct binary and never overwrites the active candidate.
`q4-residual-cache-matrices-a2.json` passes the same 108 matrix gates. The
batch-two down projection improves from the first experiment's 0.9055 ms to
0.4166 ms (production is 0.4337 ms in the second run), consistent with cache
pressure contributing to that regression. Other large batch-two/four cases
still give no compelling general improvement. Both compiled variants have no
local-memory spills (40 registers for rows 2/3/4, 48 for row 1 in the first
variant). The model experiment retains the original measured single-row scope;
no batched production selector is added based on these results.

The isolated candidate's full A/B/B/A HTTP comparison passes in
`mistral-spm-q4-residual-http-a2.json`: all generated text matches the ordinary
native Q4 path, and every phase passes host restore, Mistral↔Qwen3 swapping,
disk reload, cancellation recovery and streamed/concurrent chat. Both use the
same Q4 file and mixed policy; only the explicitly reported decode arithmetic
changes. Six measured rounds per workload per phase, after three warmups:

| Input / concurrency | Ordinary native Q4 | Integer-dot candidate |
| --- | ---: | ---: |
| 16 / 1 | 0.716963 s | 0.584562 s |
| 129 / 1 | 1.028997 s | 0.895928 s |
| 400 / 1 | 1.246418 s | 1.110303 s |
| 129 / 2 | 1.520831 s | 1.508158 s |

Single-request latency improves 11–18%; the unchanged batch-two path is
approximately tied. These are dllm-versus-dllm results, not new llama.cpp parity
claims. Normal per-model dispatch requires separate confirmation from this
validation-only server entry point.

The normal opt-in implementation lives in `native/q4_residual.cu` and
`dllm/native_q4_residual.py`, CMake target `dllm-q4-residual`, ABI 201. Set
`DLLM_Q4_RESIDUAL_LIBRARY` before creating the executor. Selection is per model,
not a global monkey patch, and bound to its library path for graph lifetime.
It requires CUDA, Q4_0, mixed precision, context <=512, Llama metadata with
32 layers, hidden 4096, intermediate 14336, 32 query/8 KV heads, head dimension
128 and vocabulary 32000. Only one-row projections with width >=4096 divisible
by 256 and outputs >=1024 use it; all other cases retain existing dispatch.
The native API itself rejects multiple rows.

The final library's 36 matrix checks pass
(`q4-residual-production-matrices-a2.json`), along with 64 initial independent
HF logit comparisons (worst RMS 0.2233%, exact greedy agreement) and all 128
runtime-layout comparisons. Reports carry the final library hash and selected
`+cuda-q4-residual` backend. All 96 extended independent checks also pass
(worst RMS 0.4182%, exact greedy agreement), matching the isolated candidate.
Normal-server confirmation passes in
`mistral-spm-q4-residual-production-http-a2.json`: both phases use the normal
CLI, backend assertions confirm selection, and every output and lifecycle check
passes. Combined A/B/B/A baseline/candidate median seconds are 0.7180/0.5875
(16 input tokens, concurrency 1), 1.0301/0.8947 (129/1), 1.2459/1.1149
(400/1), and 1.5222/1.5172 (129/2). Thus the production opt-in preserves the
11–18% single-request latency reduction, with the unchanged batch-two path
approximately tied. These remain comparisons within dllm, not llama.cpp parity.
The four new CPU regression tests cover
ABI rejection, CPU-buffer rejection, opt-in/topology/precision/context gates,
per-executor path binding and unchanged dense/batch fallback. The complete
local suite passed 302 tests before the separate admission-experiment tests.

Prompt-processing profiling also retains a rejected broad expansion of the
existing paired-dot Q4 kernel in `q4-prefill-tiles-a2.json`: 48 tile/matrix
checks pass; 12 BK256 variants exceed the A2 shared-memory limit and are
retained as failures. Modest 129-row wins become regressions at 258/400 rows. No
prefill dispatch is changed from those results. A distinct single-dot layout
is prepared for separate measurement; bounded idle-group admission is tracked in #47.

The complete local suite passes 306 tests, including four deterministic FIFO,
cancellation, deadline and wakeup checks for the isolated #47 admission wrapper.


### Published TinyLlama positional SentencePiece

The unchanged publisher file `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, revision
`52e7645ba7c309695bec7ac98f4f005b139cf465`, file
`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, has SHA256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`
and 668,788,096 bytes. Its ordinary token scores are all zero. The previous
static-rank adapter correctly rejected it: resolving equal scores by token
rank changes the file's native positional merge behavior.

`PositionalSentencePieceTokenizer` now reconstructs an identity-normalized BPE
model from GGUF metadata and uses the standalone native C++ SentencePiece
library for score ties. Explicit special-token partitioning and prefix policy
preserve GGUF semantics; Rust handles vocabulary lookup and byte decoding.
Serialized state retains the native model, including through checkpoint load.
The optional GGUF extra includes SentencePiece 0.2.2 and protobuf; no external
inference engine or original tokenizer sidecar is used. Arbitrary normalizers,
Unigram, positional unused-token semantics and structured-output integration
remain outside the qualified scope and fail explicitly.

`tinyllama-q4-k-m-tokenizer-reference.json` passes the existing expanded corpus:
512 encodings, 1,024 full decodings and 512 generated continuations. A second
seed-9167 adversarial corpus (`tinyllama-positional-requests.jsonl`) combines
overlapping ordinary pieces, explicit controls, repeated whitespace, Unicode,
combining marks and NUL. Its captured native oracle and
`tinyllama-positional-reference.json` pass another 1,536 encodings, 3,072 full
decodings and 1,536 continuations. Both verify serialization round trips against
the unchanged pinned llama.cpp public vocabulary API. These checks follow the
published GGUF scores, not the original HF tokenizer's different scores.

Initial independent model accuracy passes all 64 checks in
`tinyllama-q4-k-m-reference-a2.json`: maximum absolute error 0.04296875,
relative RMS 0.32765%, exact greedy agreement. Independent dequantization uses
`gguf.quants`, with HF eager Llama execution configured from
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. The file contains 135 Q4_K,
21 Q6_K and 45 F32 tensors, with 22 layers, hidden width 2048, intermediate
width 5632, 32 query heads and 4 KV heads. Testing uses the A2, mixed FP16
matrix/FP32 residual precision, native K-quant decode and context 512.
All 96 extended independent checks also pass in
`tinyllama-q4-k-m-reference-extended-a2.json` (maximum absolute error 0.0546875,
relative RMS 0.46157%, exact greedy agreement). All 128 execution-layout checks
pass in `tinyllama-q4-k-m-layout-a2.json` (worst relative RMS 0.09839%), covering
batches 1/2/4, eager/graphs, fixed/paged KV and chunked prefill. Both reports
record the native K-quant library hash. The local suite passes 307 tests.

The same published file passes all output and lifecycle checks in
`tinyllama-q4-k-m-http-a2.json`: A/B/B/A phases, three warmup and six measured
rounds per phase, 16 output tokens, context 512 per slot, four slots and FP16 KV.
Both dllm phases pass host restore, disk reload, cancellation recovery,
streaming/concurrent chat and a real TinyLlama -> Qwen3 -> TinyLlama swap.
All measured generated texts match the pinned llama.cpp server. Combined median
seconds (dllm / llama.cpp) are:

| Input tokens / concurrency | dllm | llama.cpp |
|---|---:|---:|
| 16 / 1 | 0.2064 | 0.1297 |
| 129 / 1 | 0.2336 | 0.1577 |
| 400 / 1 | 0.2740 | 0.2180 |
| 129 / 2 | 0.3414 | 0.1905 |
| 129 / 4 | 0.4940 | 0.3185 |

Thus serving compatibility passes for these workloads, while dllm remains
26–79% slower. Matching output does not establish performance parity or qualify
long contexts, arbitrary SPM checkpoints or structured generation.

The distinct single-dot prefill sweep in `q4-prefill-single-dot-a2.json` is
rejected for performance: even its fastest passing large-matrix tiles are
roughly 5–7 times slower than production expansion plus dense GEMM. Tile
compilation failures remain in the report. No prefill dispatch changes follow
from this experiment.


### Bounded idle admission experiment

`mistral-spm-q4-admission-1ms-a2.json` passes output agreement and every
host/disk restore, cross-model swap, cancellation and chat lifecycle check.
This uses the original native Q4 kernel in both engines; only the isolated
idle-group wait changes. It does not include the residual-kernel speedup.
A/B/B/A combined median total seconds (ordinary / 1 ms admission) are:

| Input tokens / concurrency | Ordinary | 1 ms admission |
|---|---:|---:|
| 16 / 1 | 0.7171 | 0.7216 |
| 129 / 1 | 1.0297 | 1.0354 |
| 400 / 1 | 1.2453 | 1.2545 |
| 129 / 2 | 1.5182 | 1.2527 |
| 129 / 4 | 1.8034 | 1.8124 |

Two-request latency improves 17.5%; four-request total latency is roughly tied.
Single-request first nonempty output medians change from 57.72 to 58.82 ms,
360.96 to 362.05 ms, and 567.26 to 571.65 ms for the three input lengths.
Do not interpret this as a uniform improvement in time to first output: at
concurrency two, the median earliest/latest first outputs within each round
change from 365.01/763.97 ms to 480.42/481.23 ms. Coalesced prefill delays the
first request while bringing forward the second. At concurrency four the
latest-first-output median stays about 1.01 seconds; 1 ms does not consistently
collect all four arrivals. These are HTTP output-chunk timings, not internal
per-token GPU latency.

The normal scheduler is unchanged. The smaller-model tradeoff is measured
separately before considering a production option. CPU coverage now includes
real condition-variable wakeups for arrival, another-model barriers,
cancellation and shutdown, alongside the deterministic deadline/FIFO checks.
The full local suite passes 309 tests, including the positional tokenizer's
longest-special-first overlap regression. Both TinyLlama native oracle corpora
pass again after that refinement; reports now include the native tokenizer
version, reconstructed model hash and adapter hash.


The smaller published TinyLlama checkpoint also passes the isolated admission
comparison in `tinyllama-q4-k-m-admission-1ms-a2.json`, including every lifecycle
check and identical outputs. Ordinary / candidate median seconds are
0.2051/0.2073 (16/1), 0.2345/0.2341 (129/1), 0.2739/0.2753 (400/1),
0.3433/0.2965 (129/2), and 0.4910/0.4931 (129/4). Two-request latency improves
13.6%; single-request first nonempty output medians increase by 1.1–1.3 ms.
Four-request total latency remains tied. Both models therefore support an
explicit latency/throughput option, not a changed default.

Normal CLI plumbing now exposes `dllm serve --batch-wait-ms 1`, bounded to
finite values from 0 through 10, with default 0. The shared helper runs once
before an idle model group begins; it is never inserted into active decode
iterations. Existing queue order, management/model barriers, cancellation and
shutdown notifications remain authoritative. CPU integration verifies option
bounds and one call per group across multiple decode steps. The full local
suite passes 313 tests. Normal-command A/B/B/A confirmation on both checkpoints
is recorded separately when complete.


Final normal-command confirmation passes on both checkpoints in
`mistral-spm-q4-admission-production-1ms-a2.json` and
`tinyllama-q4-k-m-admission-production-1ms-a2.json`. All four phases invoke
`python -m dllm serve`; recorded arguments select 0/1/1/0 ms. Every generated
text and lifecycle gate passes. Combined median total seconds are:

| Input / concurrency | Mistral default | Mistral 1 ms | TinyLlama default | TinyLlama 1 ms |
|---|---:|---:|---:|---:|
| 16 / 1 | 0.7140 | 0.7213 | 0.2089 | 0.2083 |
| 129 / 1 | 1.0257 | 1.0351 | 0.2341 | 0.2348 |
| 400 / 1 | 1.2405 | 1.2536 | 0.2745 | 0.2752 |
| 129 / 2 | 1.5256 | 1.2518 | 0.3441 | 0.2954 |
| 129 / 4 | 1.7985 | 1.8078 | 0.4937 | 0.4941 |

The normal option reduces two-request latency by 17.9% and 14.1%, respectively.
Four-request total latency remains tied. Single-request first nonempty output
median differences are 1.09/2.59/6.93 ms for Mistral and 0.92/1.11/1.06 ms for
TinyLlama across 16/129/400-token inputs. These are observed end-to-end
measurements, not a claim that all variation is caused by the 1 ms timer.
The option remains disabled by default. Context, kernels, precision and weights
are identical within each comparison; this is not a new cross-engine parity
claim. The implementation and validation complete the bounded opt-in scope of
issue #47; wider scheduling/performance work remains open.


### TinyLlama full context and activation-precision diagnostic

The ordinary native K-quant path passes all 64 independent HF checks at context
2048 in `tinyllama-q4-k-m-reference-2048-a2.json`, including a 2041-token prompt
and eight output positions. Worst RMS is 0.32765%, maximum absolute error
0.04296875, and every greedy token agrees. All 128 layout checks also pass in
`tinyllama-q4-k-m-layout-2048-a2.json` (worst RMS 0.09839%), using prompt lengths
4/128/1024/1920, batches 1/2/4, eager/graphs, fixed/paged KV and 128-token chunks.
The layout verifier now accepts `--context`; its 512 default retains the original
workloads. These checks qualify the checkpoint's full declared context for
those execution layouts; existing HTTP performance/lifecycle measurements
remain at 512.

The pinned llama.cpp CUDA source converts inputs to Q8_1 in its
[`mmvq` path](https://github.com/ggml-org/llama.cpp/blob/836d57176dc699a726c55418e4f96b8ca628e1bf/ggml/src/ggml-cuda/mmvq.cu),
while dllm's qualified Q4 residual path uses two int8 terms. Same weight bytes
and FP16 KV do not imply identical internal activation arithmetic. An independent
single-term absmax/127 diagnostic uses legacy Q4_0 matrices with Mistral-sized
projections. It is enabled only when building
`scripts/experiments/q4_residual.cu` with `-DDLLM_Q4_SINGLE_TERM=1`. It uses FP32
activation scaling and is not a bitwise reimplementation of Q8_1.

`q4-single-term-matrices-a2.json` retains the failed numerical gate: 18 of 108
matrix/range checks pass, 90 fail the unchanged 0.1% RMS limit; worst relative
RMS is 0.56752%. All timings are diagnostic, not accepted model-performance
claims. In the same run, large single-row cases are 13–15% faster than the
qualified two-term production kernel, whose hash is also recorded:

| Outputs / width | Original native ms | Qualified two-term ms | Diagnostic one-term ms |
|---|---:|---:|---:|
| 6144 / 4096 | 0.1500 | 0.1189 | 0.1034 |
| 4096 / 4096 | 0.1015 | 0.0817 | 0.0708 |
| 28672 / 4096 | 0.6966 | 0.5329 | 0.4542 |
| 4096 / 14336 | 0.3426 | 0.2823 | 0.2397 |
| 32000 / 4096 | 0.7801 | 0.5959 | 0.5071 |

This identifies a measurable accuracy/cost tradeoff; it does not attribute the
entire cross-engine gap to that difference. The diagnostic is not selected by
normal runtime dispatch and does not replace the qualified two-term kernel.
Build with `nvcc -O3 -std=c++17 -arch=sm_86 -DDLLM_Q4_SINGLE_TERM=1 -shared
-Xcompiler -fPIC scripts/experiments/q4_residual.cu -o PATH.so`, then use
`profile_q4_residual.py --two-term-library PATH_TO_PRODUCTION_LIBRARY` with the
existing experimental/native library environment variables. Its nonzero exit
status is the preserved accuracy failure, not a successful qualification.


### TinyLlama native residual decode at 512 context

The existing opt-in `DLLM_KQUANT_RESIDUAL_LIBRARY` passes 64 initial and 96
extended independent HF checks on this published checkpoint, with exact greedy
agreement. Reports `tinyllama-q4-k-m-residual-reference-a2.json` and
`tinyllama-q4-k-m-residual-reference-extended-a2.json` record the selected normal
backend and library hash. Worst RMS is 0.44765% initially and 0.46080% in the
extended suite. All 128 layout checks pass in
`tinyllama-q4-k-m-residual-layout-a2.json` (worst RMS 0.10958%).

Normal-CLI A/B/B/A serving also passes every output/lifecycle gate in
`tinyllama-q4-k-m-residual-production-http-a2.json`, including actual cross-model
swaps. Combined baseline/residual median seconds are 0.2084/0.1906 (16/1),
0.2342/0.2161 (129/1), 0.2741/0.2584 (400/1), 0.3431/0.3397 (129/2), and
0.4937/0.4893 (129/4). Single-request latency improves 5.7–8.5%; the unchanged
multi-row decode path is approximately tied. This run keeps idle admission at
its default zero in both engines and does not claim combined-policy results.

This initial qualification bounded residual selection to context 512.
The subsequent TinyLlama-specific extension is documented below.


### Normal TinyLlama residual selection through 2048

Isolated long-context replay first passes 160 independent checks and 128 layout
checks in `tinyllama-q4-k-m-residual-experimental-*-2048-a2.json`. Production
selection is then extended only for the tested TinyLlama topology: 22 layers,
hidden 2048, intermediate 5632, 32 query/4 KV heads, head dimension 64,
vocabulary 32000, declared context 2048, unscaled RoPE with theta 10000 and no
sliding window. Other Llama models keep the 512-context limit. Q5, other
precisions/devices, multi-row decode and non-2048-wide projections retain their
existing exclusions or fallback. The CUDA library itself is unchanged.

The final normal selection passes all 160 independent checks and all 128 layout
checks in `tinyllama-q4-k-m-residual-production-reference-2048-a2.json`,
`tinyllama-q4-k-m-residual-production-reference-extended-2048-a2.json` and
`tinyllama-q4-k-m-residual-production-layout-2048-a2.json`. Normal backend
selection and library hashes are recorded. Maximum relative RMS remains
0.44765% initially, 0.46080% extended, and 0.10958% across layouts, with exact
greedy agreement. CPU regression coverage rejects adjacent unqualified
configurations; the full local suite passes 314 tests.

The lab previously had SentencePiece 0.2.0 installed; earlier reports retain
that provenance. The declared dependency is 0.2.2. After aligning the lab with
that pin, both complete tokenizer corpora pass again in
`tinyllama-positional-spm022-reference.json` and
`tinyllama-q4-k-m-tokenizer-spm022-reference.json` (2,048 encodings, 4,096 full
decodings and 2,048 continuations). The final normal model/layout reports record
SentencePiece 0.2.2 and protobuf 4.25.8 explicitly. No vocabulary metadata or
expected oracle outputs changed during the upgrade.

The engine harness now accepts `--context` and `--long-prompt-tokens`, preserving
its original 512/400 defaults. A normal-command server comparison uses context
2048 and a 2032-token prompt, leaving exactly 16 output tokens of capacity.
The completed normal-CLI report is
`tinyllama-q4-k-m-residual-production-http-2048-ocean-a2.json`. All four
A/B/B/A phases pass exact generated-text equality and host restore, disk reload,
cancellation recovery, concurrent/streaming chat and cross-model swapping.
Both paths use FP16 KV and the unchanged packed checkpoint; admission stays zero.

| Input tokens / concurrency | Original median seconds | Residual median seconds |
| --- | ---: | ---: |
| 16 / 1 | 0.207586 | 0.190698 |
| 129 / 1 | 0.233350 | 0.218965 |
| 2032 / 1 | 0.543438 | 0.528967 |
| 129 / 2 | 0.341526 | 0.340122 |
| 129 / 4 | 0.492558 | 0.494805 |

These medians pool six measured rounds from each of two phases after three
warmups per case. Single-request latency falls 2.7–8.1%; concurrent paths are
approximately tied. This compares dllm implementations, not llama.cpp.

The original binary-search corpus naturally stopped after five output tokens
at the near-limit prompt. Its failed fixed-16-token eligibility report is
preserved as `tinyllama-q4-k-m-residual-http-2048-short-output-control-a2.json`.
The harness now accepts and records `--prompt-text`, retaining its old default.
An ocean/weather corpus was the first alternative preflighted to produce all
16 tokens at every tested input length, before candidate timing. EOS handling
and equality/length gates were not changed. These synthetic repeated-token
prompts measure runtime behavior, not answer quality; no latency comparison
across different prompt corpora is claimed.


### TinyLlama same-file llama.cpp comparison at 2048

`tinyllama-q4-k-m-llama-http-2048-ocean-a2.json` preserves a failed output
equality gate. Every run completed the requested 16 output tokens, but the
2032-token prompt diverges near the end, and llama.cpp produces two distinct
texts across the four-request workload. dllm is consistent across those runs.
All dllm lifecycle checks pass. This is not an output-equivalent benchmark win.

| Input tokens / concurrency | dllm median seconds | llama.cpp median seconds |
| --- | ---: | ---: |
| 16 / 1 | 0.207917 | 0.128954 |
| 129 / 1 | 0.233868 | 0.157290 |
| 2032 / 1 | 0.543775 | 0.588973 |
| 129 / 2 | 0.341767 | 0.194465 |
| 129 / 4 | 0.492808 | 0.321245 |

The same unchanged checkpoint, prompt IDs, output lengths, FP16 KV, context
capacity and A/B/B/A protocol are recorded in the report. These timings retain
useful diagnostic evidence, but the long and four-request outputs differ.
`tinyllama-q4-k-m-ocean-reference-2048-a2.json` then independently replays all
three exact prompt sequences against HF with independently dequantized,
FP16-rounded coefficients: all 96 logit checks pass, every greedy token agrees,
and eager/graph paths agree. Worst relative RMS is 0.73287%, maximum absolute
error 0.072265625. This validates dllm against its declared arithmetic reference;
it does not establish bitwise cross-engine arithmetic or general answer quality.


### TinyLlama projection and short-prefill probes

`tinyllama-kquant-projections-a2.jsonl` and
`tinyllama-kquant-short-prefill-a2.jsonl` use the unchanged published checkpoint
and `scripts/profile_kquant.py`. All matrix results pass the existing 0.1%
relative RMS gate against resident FP16-expanded coefficients. Timings are CUDA
event measurements of ten graph replays after warmup; these isolated probes
are not HTTP or full-model performance claims.

At 129 rows, the tested fused packed tiles are slower than expansion plus GEMM
for all five projections. At 16/32 rows, the best existing tile improves the
layer-zero gate/up projection by roughly 16–17%, while QKV is tied or slower.
At 64 rows, all tested fused tiles regress. `--prefill-rows` now permits these
explicit prefill lengths without also rerunning decode tuning. The head's
multi-row probes are synthetic matrix cases: normal prefill projects only the
last hidden row, so those head timings must not be added to model latency.
These findings do not change production dispatch. Expansion tile efficiency is
being measured separately before choosing a candidate for model qualification.


`tinyllama-kquant-expand-tiles-a2.json` isolates the expansion kernel on those
same five packed projections. All 60 tile variants produce exactly equal FP16
coefficient tensors. A common 8-row / 512-column tile with four warps reduces
expansion-only GPU time by roughly 15–20% versus the current 4x256 tile (the
QKV-specific fastest tile is 16x256, but 8x512 is close). This avoids separate
per-projection tuning and changes neither arithmetic nor persistent storage.
The experiment records checkpoint/kernel hashes and Torch/Triton/GPU identity.
`profile_kquant_expand.py` reproduces it; full-model/server qualification is
separate. No production dispatch change follows from these matrix timings alone.


The explicit validation-only override in `scripts/experiments/kquant_expand.py`
then passes 96 independent model checks on the exact ocean benchmark prompts at
2048 context, with exact greedy agreement and the same 0.73287% maximum RMS as
the original expansion. It also passes all 128 fixed/paged, eager/graph,
batched/chunked layout comparisons. Reports are
`tinyllama-expand-tiles-reference-a2.json` and
`tinyllama-expand-tiles-layout-a2.json`; both record the override and its source
hash. The hook preserves fallback for other widths, formats and precisions.
The benchmark's `--expand-tiles` option selects an explicit experimental server
module only in candidate phases; normal CLI dispatch remains unchanged.


`tinyllama-expand-tiles-http-a2.json` completes the isolated A/B/B/A server
comparison at 2048 context. Every output and lifecycle gate passes. Original /
experimental pooled median seconds are 0.204434/0.203463 (16/1),
0.232034/0.229498 (129/1), 0.545350/0.541215 (2032/1),
0.340064/0.336078 (129/2), and 0.492983/0.488826 (129/4).
Observed request-latency gains are only 0.5–1.2%, despite the larger isolated
expansion gain. Long-prompt baseline medians also vary from 0.541503 to 0.546988
between phases. This remains an experiment: no production option or default
change is justified by a single small effect. The normal decode path is the
same in both sides, and this does not measure stacking with residual decode.


### TinyLlama concurrent decode bottleneck and launch geometry

`tinyllama-decode-b2-a2.json` and `tinyllama-decode-b4-a2.json` profile ordinary
native K-quant decode at context 512. Packed matrix kernels account for 11.635
of 12.229 ms GPU time per two-request decode step (95.15%), and 17.953 of
18.739 ms per four-request step (95.81%). Profiling overhead makes these
bottleneck measurements distinct from unprofiled HTTP latency.

`tinyllama-kquant-launch-a2.json` compares the installed binary with separately
compiled 64/128/256-thread blocks, on five actual checkpoint projections and
one/two/four input rows. Every one of 120 comparisons preserves output bits,
including bias; forward/reverse library order brackets the candidates with the
original control. The rebuilt 128-thread control is approximately tied. The
other geometries offer no broad improvement and sometimes regress by 1–3%.
Default launch geometry remains 128 threads. Reproduce with
`scripts/profile_kquant_launch.py`; compile isolated variants using
`-DDLLM_KQUANT_THREADS=64` or `256`, retaining `--fmad=false` and `sm_86`.
These are matrix experiments, not new qualified model or HTTP paths.


`tinyllama-kquant-cache-policy-a2.json` retains a second native comparison:
ordinary weight loads versus explicit L1-bypassing loads. All 90 actual-matrix
checks preserve output bits. Cache bypass is slower by roughly 3–16% across
these cases, so it is not selected. The rebuilt unflagged control is roughly
tied with the installed binary. `DLLM_KQUANT_BYPASS_L1` is a compile-time
experiment only; normal builds retain ordinary loads and 128-thread blocks.
`kquant-rebuilt-control-coefficients-a2.json` independently probes the rebuilt
unflagged library against GGUF dequantization, including mixed segment alignment
and tail rows, with all coefficient checks passing. The installed library is
unchanged. No runtime setting or performance improvement is claimed from either
launch-geometry or cache-policy sweep.


### Resident projection-cache prototype and reload workspace fix

`scripts/experiments/kquant_dense_cache.py` explores a separate memory/speed
tradeoff: retain FP16-expanded projection copies for multi-row calls while
keeping authoritative packed weights and single-row packed decode. The cache
belongs to each executor's Linear object, is capped at 3 GiB, and falls back
when the requested cache exceeds the cap. It is validation-only: production
model admission does not yet account for this additional residency. It has no
normal CLI option and is not a new default.

TinyLlama caches 2,068,840,448 additional bytes (about 1.927 GiB). Initial
`tinyllama-dense-cache-reference-a2.json` and
`tinyllama-dense-cache-layout-a2.json` pass 96 independent exact-prompt checks
and 128 layout comparisons at 2048 context. Maximum RMS is 0.73287% against HF
and 0.10232% across layouts, with greedy agreement.

The first unload proof fails despite all cache tensor references being released:
`tinyllama-dense-cache-release-initial-failure-a2.json`. A separate ordinary
dense control (independent CPU coefficient expansion, no cache override) also
retains extra allocated memory after successive executor unloads: 34,078,720
then 51,118,080 bytes. Diagnostic-only cuBLAS workspace clearing releases all
of those bytes; the failed controls and `*.workspace.json` reports remain
preserved. This establishes stream-associated workspace retention, not an
unbounded model-tensor leak. The private diagnostic operation is never used by
the runtime.

Dense and batched dense warmups now share one serialized CUDA stream per device.
Caller-stream dependencies are maintained on normal and exceptional exits.
`tinyllama-dense-control-release-fixed-a2.json` passes three ordinary dense
activation/unload cycles, each returning to 25,559,040 bytes (24.375 MiB) of
reusable workspace. `tinyllama-dense-cache-release-fixed-a2.json` passes two
cached cycles plus a one-byte-budget fallback with the same stable allocation;
all executor, weight and cache references are released. The full local suite
passes 317 tests, including stream reuse, device isolation, exception handling
and overlapping warmup serialization. See GitHub issue #48. Other independently
implemented architecture warmup paths are outside this fix's scope.


Post-fix `tinyllama-dense-cache-warmup-reference-a2.json` and
`tinyllama-dense-cache-warmup-layout-a2.json` repeat all 96 independent model and
128 layout checks successfully, with unchanged maximum errors. These checks use
stream reuse without clearing cuBLAS workspaces. HTTP timing and hot-swap proof
for the prototype are evaluated separately before any production cache decision.


The completed isolated cache HTTP run, `tinyllama-dense-cache-http-a2.json`,
passes all output and lifecycle gates, including actual TinyLlama/Qwen3 model
swaps, disk/host reloads, cancellation, streaming and concurrent chat. Both
sides include the warmup-stream fix. Original/cached pooled medians are
0.208453/0.191185 seconds (16/1), 0.234636/0.214548 (129/1),
0.545332/0.524106 (2032/1), 0.343211/0.330513 (129/2), and
0.495083/0.387239 (129/4). Four-request latency falls 21.8%; single-request
latency falls 3.9–8.6%. These are within-dllm gains for an additional 1.927 GiB
of residency, not cross-engine wins or a free memory optimization. The
prototype's performance proof is separate from normal-path admission and
status qualification, which must pass before the cache is supported.


### Normal admission-accounted TinyLlama resident cache

The normal executor now supports `DLLM_KQUANT_DENSE_CACHE_MIB`, default zero.
Selection is bounded to the qualified TinyLlama metadata, mixed precision,
CUDA Q4_K/Q6_K and context up to 2048. Core weights, KV and existing workspace
reserves are accounted first. A complete set of eligible projections is cached
only when both the requested per-executor budget and remaining free VRAM cover
it; otherwise ordinary packed execution remains available. The authoritative
packed weights remain resident. Aliased storage is counted once, embedding
storage is excluded, and single-row decode keeps its existing path.

A 2048 MiB budget admits the tested 1973 MiB cache. Backend status includes
`resident-kquant-cache`; `/v1/models/status` reports
`gpu_projection_cache_bytes` per loaded model. The setting does not change
other architectures or promote the experimental cache globally.

`tinyllama-dense-cache-production-reference-a2.json` and
`tinyllama-dense-cache-production-layout-a2.json` pass 96 independent and 128
layout checks with assertions that normal cache selection is active, reporting
2,068,840,448 bytes. `tinyllama-dense-cache-production-release-a2.json` passes
two normal cached unloads plus a disabled-cache control, with all tensor
references released and 25,559,040 bytes of stable cuBLAS workspace. The full
local suite passes 320 tests, including exact budget boundaries, insufficient
free memory, unique-storage accounting, scope exclusions and preservation of
single-row decode. `tinyllama-dense-cache-production-budget-fallback-a2.json` additionally checks
a positive but insufficient 1 MiB cache budget: zero projections are retained,
ordinary execution succeeds and all memory-release gates pass.

The normal-CLI A/B/B/A confirmation in
`tinyllama-dense-cache-production-http-a2.json` passes every output and lifecycle
gate, verifies the normal backend tag, and checks reported cache residency:
zero in control phases and exactly 2,068,840,448 bytes in candidate phases.

| Input / concurrency | Original seconds | Cached seconds | Original output tokens/s | Cached output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 16 / 1 | 0.208638 | 0.191722 | 76.69 | 83.45 |
| 129 / 1 | 0.233278 | 0.212179 | 68.59 | 75.41 |
| 2032 / 1 | 0.542663 | 0.521587 | 29.48 | 30.68 |
| 129 / 2 | 0.341168 | 0.329117 | 93.80 | 97.23 |
| 129 / 4 | 0.488414 | 0.390238 | 131.04 | 164.00 |

Latencies and aggregate output throughput pool the measured rounds from both
phases of each engine, excluding warmups. Throughput includes request prefill;
it is not decode-only token speed. Four-request latency falls 20.1% and aggregate
output throughput rises 25.2%. Single-request latency falls 3.9–9.0%. Both sides
use ordinary native decode and zero idle coalescing. Combined-option results
are qualified separately; these are within-dllm gains, not llama.cpp parity.
The additional GPU-resident copies cost 1973 MiB in total across all cached
projections; the cache does not reduce the authoritative packed checkpoint.


### Resident cache combined with residual decode

`tinyllama-cache-residual-{reference,reference-extended,layout}-a2.json`
qualify the normal cache together with the native residual decoder: 96 exact
benchmark-prompt, 96 extended independent-reference and 128 layout checks pass.
Worst relative RMS is 0.8816%, 0.4608% and 0.1017%, respectively, with exact
greedy agreement. The same 1973 MiB additional cache budget is consumed.

`tinyllama-cache-residual-production-http-a2.json` compares normal CLI phases
in A/B/B/A order. Both sides use residual decode; only the candidate enables
the resident cache. All generated outputs and lifecycle checks pass.

| Input / concurrency | Residual only seconds | With cache seconds | Residual only output tokens/s | With cache output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 16 / 1 | 0.192022 | 0.176063 | 83.33 | 90.88 |
| 129 / 1 | 0.217604 | 0.198049 | 73.53 | 80.79 |
| 2032 / 1 | 0.528808 | 0.506584 | 30.26 | 31.58 |
| 129 / 2 | 0.340689 | 0.325770 | 93.93 | 98.23 |
| 129 / 4 | 0.489085 | 0.384674 | 130.86 | 166.38 |

Four-request latency falls 21.3%, with 27.1% higher aggregate throughput.
These pooled medians include prefill and exclude warmups. Idle coalescing is
zero on both sides; this does not qualify every option together or establish
a cross-engine win.

The harness now retains end-to-end lifecycle request timing separately from
steady-state rounds. Cached host restores took 1.135/1.244 seconds and switches
back from Qwen3 took 1.313/1.583 seconds; disk reload requests took 2.304/2.956
seconds. These include activation and generation, with only two observations
per action. Filesystem caches were not flushed. They prove usable reload paths,
not a statistically established hot-swap improvement; control disk reloads
were faster in this small sample (2.145/2.356 seconds).

### Larger Qwen3-1.7B K-quant qualification

The cached HF checkpoint `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` was exported as F16, then quantized
as Q4_K_M using the unchanged reference tooling. The converter matches revision
`836d57176dc699a726c55418e4f96b8ca628e1bf`, SHA256
`e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f`.
The quantizer binary SHA256 is
`6718881be431d1ed3539ccec5a98f2adb04b89ae2e29d0b70b016a73e5920202`.
The resulting `qwen3-medium-import-q4_k_m.gguf` SHA256 is
`d8a2811d8568778c79219100c92fe540889d54e1dbb2d6e5ed9c1c44d2b3a55d`.
It contains 169 Q4_K, 29 Q6_K and 113 F32 tensors. This is a generated fixture,
not a claim that every published Qwen GGUF behaves identically.

At 512 context, `qwen3-medium-kquant-reference-512-a2.json` passes all 64
independent HF comparisons with exact greedy agreement, maximum absolute logit
error 0.1328125 and relative RMS 0.6857%. The experimental resident-cache run
`qwen3-medium-cache-reference-512-a2.json` passes the same gates. Both use the
standard Qwen3 policy: FP16 matrices with FP32 residual/nonlinear operations.
The independent reference runs first and releases GPU residency before dllm
executes; its logits remain available for comparison.

The broader validation-only cache admits 3,440,902,144 bytes under its 4 GiB
cap. `qwen3-medium-cache-layout-512-a2.json` passes all 128 fixed/paged,
eager/graph, batch 1/2/4 and chunked-prefill comparisons, worst relative RMS
0.2716%. This is a new model/shape qualification, not a long-context or
production-cache qualification. The subsequent normal-path qualification is
recorded below; these initial prototype checks alone did not establish it.

`qwen3-medium-cache-reference-extended-512-a2.json` additionally passes 96
independent comparisons over multi-turn, Unicode, code and longer raw prompts;
maximum absolute error is 0.1484375, relative RMS 0.7620%, with exact greedy
agreement. No numerical tolerance was changed.

The experimental A/B/B/A real-server comparison
`qwen3-medium-cache-http-512-a2.json` passes every generated-output and lifecycle
gate, including actual 1.7B/0.6B switching, host restore, disk reload,
cancellation recovery, streaming and concurrent chat.

| Input / concurrency | Original seconds | Cached seconds | Original output tokens/s | Cached output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 16 / 1 | 0.330635 | 0.308053 | 48.39 | 51.94 |
| 129 / 1 | 0.368352 | 0.342680 | 43.44 | 46.69 |
| 400 / 1 | 0.426845 | 0.397954 | 37.48 | 40.21 |
| 129 / 2 | 0.533733 | 0.511980 | 59.96 | 62.50 |
| 129 / 4 | 0.756703 | 0.593068 | 84.58 | 107.91 |

Medians pool measured rounds from both phases, excluding warmups. Throughput
includes prefill. Four-request latency falls 21.6%, single-request latency
6.8–7.0%; this is within-dllm performance with additional residency, not a
cross-engine comparison. Cached host-restore requests took 1.535/1.712 seconds,
switches back 1.800/1.973 seconds and disk-reload requests 3.770/5.009 seconds.
Only two observations per action are available; no hot-swap speedup is claimed.

The ordinary and experimental cache paths also pass their separate 64-check
independent model suites at 2048 context:
`qwen3-medium-{kquant,cache}-reference-2048-a2.json`. Worst relative RMS remains
0.6857%, maximum absolute error 0.1328125 and all greedy decisions agree.
`qwen3-medium-cache-release-a2.json` passes two 4 GiB-budget cached activations
and a positive-but-insufficient 1 MiB fallback. All executor/weight/cache weak
references expire after each unload; allocated GPU memory returns to exactly
25,559,040 bytes each time. This retained workspace is stable across all three
cycles and the cached model tensors are released.

### Normal Qwen3-1.7B resident cache through 2048

Normal selection now includes the tested Qwen3-1.7B geometry, standard Qwen3
precision, CUDA FP16 matrix storage and Q4_K/Q6_K projections through context
2048. `DLLM_KQUANT_DENSE_CACHE_MIB=4096` admits 3,440,902,144 bytes (3281.5 MiB).
It uses the same all-or-nothing free-memory/budget accounting and unchanged
single-row decode as TinyLlama. Other Qwen geometries, longer contexts and
FP32 execution remain excluded. Unit coverage checks these independent scope
boundaries; the full cache-change regression suite passes 321 tests.

`qwen3-medium-cache-production-reference-2048-a2.json` passes 96 extended HF
comparisons with normal selection required, and
`qwen3-medium-cache-production-layout-2048-a2.json` passes 128 layout checks.
Maximum independent relative RMS is 0.7620%, layout RMS 0.2716%; all greedy
decisions agree. `qwen3-medium-cache-production-release-a2.json` repeats two
cached activations plus a positive insufficient-budget fallback; every model
reference is released and all three unloaded states retain the same 24.375 MiB
of CUDA workspace.

`qwen3-medium-cache-production-http-2048-a2.json` uses the normal CLI for all
A/B/B/A phases. Reported cache bytes are exactly zero/3,440,902,144/3,440,902,144/zero.
Every output, host/disk reload, cross-model swap, cancellation, streaming and
concurrent-chat gate passes.

| Input / concurrency | Original seconds | Cached seconds | Original output tokens/s | Cached output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 16 / 1 | 0.344322 | 0.301089 | 46.51 | 53.14 |
| 129 / 1 | 0.373782 | 0.340511 | 42.81 | 46.99 |
| 2032 / 1 | 0.822428 | 0.790881 | 19.45 | 20.23 |
| 129 / 2 | 0.537348 | 0.514001 | 59.55 | 62.26 |
| 129 / 4 | 0.763758 | 0.597789 | 83.80 | 107.06 |

These pooled warm-trial medians include prefill. Four-request latency falls
21.7%; single-request latency falls 3.8–12.6% in this workload. Idle coalescing
is zero on both sides. The 2048-context test is separate from the earlier
512-context prototype run; timings across those runs are not a matched
comparison. This is not a llama.cpp/vLLM performance parity claim.

### Explicit uniform-window Mistral fixture

`prepare_gguf_window_comparator.py` creates a separate validation file using
Mistral-7B-Instruct-v0.1's pinned HF configuration. The earlier reference export
omitted its window; the new file adds only `llama.attention.sliding_window=4096`.
This changes declared attention semantics and is not a claim that the original
export encoded sliding-window attention.

`mistral-window-fixture-a2.json` proves all 291 tensor representations and all
other metadata values are unchanged. Original GGUF SHA256:
`cbea665de7de1050c1741ae205422a47bd62815c84c655fba55962fdc805ed66`;
new windowed fixture SHA256:
`c320624e1b0b443e7b9c0e7449bff3780a3bc8e91e070d4c884679b85f1867bb`.
HF configuration SHA256:
`cf25cdf4719f181d1d1d371973285d9afe9afde0d0c6a6fd48de857555ce1e0d`,
from immutable revision `ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca`.

Independent validation now has an explicit `--reference-prefill-chunk-size`
option to bound HF eager-attention temporaries for a large reference model.
It preserves HF's own execution and KV state; no dllm kernels enter the oracle.
A small independent HF Mistral test compares whole/chunked prompt outputs and
continuations at lengths 15/16/17/63 around a 16-token window. The model gate
can include declared window boundaries explicitly, and layout validation can
select four prompt lengths with a paged pool sized for all admitted tokens.
Fixture preparation and test-harness checks alone do not qualify runtime support.

The independent A2 gate in `mistral-window-reference-a2.json` passes 96 logit
comparisons, including prompts of 4095/4096/4097 tokens and eight continuation
steps with eager and graph replay. Maximum absolute error is 0.0625, relative
RMS 0.2572%, and every greedy token agrees with HF eager attention. The oracle
uses 128-token prefill chunks and releases reference GPU weights before dllm
runs. dllm uses the existing FP16 Mistral arithmetic policy and native Q4 CUDA
GEMV; the separate Llama residual optimization remains unselected. Layout and
normal-service qualification are recorded separately when complete.

The initial long-window layout run is retained as
`mistral-window-layout-underbudget-a2.json`. Its new test-pool calculation
rounded the combined token count, under-reserving one page for simultaneously
advancing requests. The runtime correctly raised `PageCapacityError`. The
harness now sums each slot's rounded reservation, including all seven decode
positions: 12352 tokens rather than 12336 for this workload. A regression test
runs those reservations through the actual `PagePool`, including all slots
crossing page boundaries and release. This is a harness-budget correction;
no runtime capacity check or numerical threshold was weakened.

`mistral-window-layout-fixed-a2.json` passes all 128 corrected layout checks
with prompts of 4/4095/4096/4097 tokens, fixed and paged storage, eager and CUDA
graph execution, batches 1/2/4 and 128-token prompt chunks. Maximum relative RMS
against fixed eager execution is 0.0737%, with exact greedy agreement. The pool
has 12352 tokens, fewer than four full 4104-token slots, but enough for each
request's individually rounded pages and seven decode positions. The runtime
still stores full-context KV; this does not introduce a rolling window cache.

The first HTTP source-comparison attempt stopped on a chat-tokenizer mismatch;
its failure log is `mistral-window-runtime-tokenizer-mismatch-a2.log`.
`mistral-window-chat-diagnostic.json` shows identical rendered templates but
GGUF inserts token 28705 after the literal BOS marker, while the HF fast
tokenizer does not. The unchanged native `llama-tokenize --no-bos` matches
dllm's complete GGUF token sequence exactly (native binary SHA256 recorded).
The tokenizer was not changed to force HF chat equivalence.

The HTTP verifier retains strict source-text comparison by default. Its explicit
`--allow-tokenizer-differences` mode records both chat inputs and checks each
source's exact decoded generation trace and usage on identical raw token IDs,
while keeping exact GGUF streaming, concurrency, restore and reload output
gates. Direct execution must still match complete logits between both loaders;
independent HF mathematical correctness is established by the separate reference
report. Source chat equivalence is reported separately instead of claiming
unlike inputs should produce identical text.

An intermediate harness attempt requested raw-completion logprobs, which the
current API intentionally rejects; its failure is retained as
`mistral-window-runtime-raw-score-rejection-a2.log`. The final trace-based check
does not request those unsupported fields or change the runtime API. Boundary
requests reserve the API's full output budget; the 4097-token request therefore
uses seven output tokens at context 4104, while the preceding two use eight.

The final `mistral-window-runtime-traces-a2.json` passes all 270 exact full-logit
comparisons between the GGUF-loaded and identically quantized HF-loaded weights
executing through dllm, with eager and graph replay. These are loader/runtime
identity checks; the separate 96-check report supplies independent HF inference
validation. Exact prompt IDs and generated-token traces are retained, including
both source chat inputs, the short timing prompt and the three window boundaries.

Normal HTTP registration, source switching, host restore, disk reload,
cancellation recovery, streaming and concurrency all pass. Every checked
response matches its source-decoded trace. Boundary prompt/output counts are
4095/8, 4096/8 and 4097/7. `exact_chat` is deliberately false: rendered chat
inputs tokenize differently, and continuation decoding also preserves a leading
space on the GGUF path that HF may strip. The corresponding exact decoded-trace
and streaming gates are true; no cross-source text-parity claim is made.

Warm short raw-request medians are 0.726445 seconds for GGUF and 0.725845 for the
HF-loaded path, both within dllm, using identical prompt IDs and the same packed
values. No extra logprobs are requested in this final run. These near-equal
measurements are functional-path evidence, not an optimization or cross-engine
performance win.

`mistral-q4-prefill-dispatch-a2.json` rejects expanding legacy Q4 projections
earlier in prompt processing. All 60 arithmetic checks pass on five real Mistral
projections at 16/32/64/128/129/256 rows. For 16–128 rows, expansion plus dense
matrix multiplication takes 1.21–4.17 times the production fused-kernel time,
including expansion on every call. At 129/256 rows both paths already use the
same expansion policy and their timings agree within 0.1%. No dispatch change
is justified, and no full-model speedup is claimed.

The follow-up `mistral-q4-small-tiles-a2.json` checks smaller fused tiles instead.
All 90 independent matrix checks pass. Across five real projection shapes,
32-row tiles reduce kernel time by 20.9–25.9% at 64 input rows and 7.4–17.4%
at 129 rows. Other measured sizes retain the production choice. The isolated
candidate still needs full-model and real serving qualification; this report
alone does not justify a normal runtime change.

`mistral-q4-small-tiles-reference-a2.json` then passes independent HF execution
at context 512, including 64/129-row prompts and eager/graph continuation.
All greedy tokens agree; worst relative logit RMS is 0.161% and maximum absolute
error is 0.0390625. The experimental override changes only the measured shapes;
layout and HTTP qualification remain separate. This windowed Mistral fixture
uses FP16 matrix and residual arithmetic, as recorded by the executor. The
HTTP benchmark now derives its precision label from the imported architecture
instead of assuming every GGUF uses Llama's mixed policy.

The candidate also passes all 128 fixed/paged, eager/graph, batch and chunk
comparisons in `mistral-q4-small-tiles-layout-a2.json`, at context 512 with
prompt lengths 64/129/256/384. Greedy choices agree and the worst relative RMS
is 0.456%. The independent reference report contains 80 full-logit comparisons.
Real HTTP timing and lifecycle checks are the next gate; no production kernel
selection changes are included in these results.

`mistral-q4-small-tiles-http-a2.json` completes the isolated candidate's HTTP
A/B/B/A check at context 512, with three warmups and six measured rounds per
phase/workload. All generated text agrees across runs; streaming, concurrency,
host restore, alias switching, cancellation and disk reload pass in all phases.
Pooled measured medians (seconds; first output is measured at the HTTP client):

| Input / concurrency | Control request | Candidate request | Control first output | Candidate first output |
|---|---:|---:|---:|---:|
| 16 / 1 | 0.709752 | 0.722669 | 0.059404 | 0.059818 |
| 64 / 1 | 0.847277 | 0.820592 | 0.187257 | 0.149421 |
| 129 / 1 | 1.038214 | 1.000931 | 0.378026 | 0.332356 |
| 400 / 1 | 1.326204 | 1.341141 | 0.655848 | 0.661508 |
| 129 / 2 | 1.548572 | 1.475719 | 0.511142 | 0.515910 |
| 129 / 4 | 1.861456 | 1.830334 | 1.070583 | 1.032098 |

At 64/129 input tokens, single-request first output improves 20.2%/12.1%,
while total latency improves 3.15%/3.59%. Unchanged 16/400-row single-request
controls regress 1.1–1.8%, showing the size of run variation. Batched results
remain sensitive to scheduler grouping; no broad concurrent or cross-engine
parity claim follows. These are ordinary native Q4 decode kernels on both
sides, with no resident dense cache. Intermediate row counts still need
measurement before replacing the isolated shape-specific experiment with a
general runtime selection rule.

`mistral-q4-intermediate-tiles-a2.json` adds 33/48/65/96/130/160 rows and
passes all 90 independent matrix checks. The 32-row tile wins across all five
projection shapes at 33 rows (10.7–16.0%), 48 rows (17.3–21.3%) and 130 rows
(7.3–17.4%). Results at 65/96/160 are mixed, with regressions up to 12.3%.
The measured evidence supports investigating a 33–64-row range, but does not
support extending the 129/130 result through 160 rows or changing all larger
prompts. Production dispatch is still unchanged.

`mistral-q4-paired-expansion-a2.json` rejects a separate expansion rewrite.
Loading each packed byte once for both nibbles preserves every FP16 coefficient
bit pattern, including signed zero, on all five real projections. The best
512-byte tile reduces expansion time only 0.4–1.0%; 256-byte tiles regress
about 5–6%, and 1024-byte tiles are effectively tied. This small isolated gain
does not justify changing the production expansion path or another serving
speedup claim.

### Normal A2 short-prompt tile policy

`dllm/q4_prefill_policy.py` now selects 32-row tiles for 33–64 input rows on
the NVIDIA A2, limited to the five measured Mistral projection shapes. Device
properties are cached per CUDA device; other devices and shapes retain their
previous choice. Single-row native decode never enters this policy. The
129/130-row experiment is not part of this selection.

`mistral-q4-production-tiles-reference-a2.json` explicitly verifies that the
normal policy was selected, then passes 112 independent HF logit comparisons
at context 512, including 33/48/64-row prompts and eager/graph continuation.
Worst relative RMS is 0.161%, maximum absolute error 0.0390625, and all greedy
tokens agree. The regression suite passes 328 tests. Long-window layout and
normal-versus-previous-policy HTTP measurements remain separate gates.

`mistral-q4-production-tiles-layout-a2.json` confirms normal policy selection
and passes all 128 layout comparisons at context 4104, using prompts
33/64/4095/4097 and the established 128-token chunks. Greedy choices agree;
worst relative RMS is 0.0737%. Because 128-token chunks largely bypass the new
33–64-row launch choice, an additional 58-token chunk run targets the policy
throughout long-prompt processing, including the final 37-token chunk crossing
the 4096-token attention window.

That targeted run, `mistral-q4-production-tiles-chunk58-a2.json`, passes all
128 comparisons with exact greedy choices and worst relative RMS 0.0903%.
The normal short-tile policy remains active throughout the 58-token chunks
and the final 37-token window-crossing chunk. The final ordinary-entrypoint
HTTP comparison includes 33/48/64-token workloads plus all previous controls.

`mistral-q4-production-tiles-http-a2.json` completes that normal-entrypoint
A/B/B/A comparison. All generated text agrees; every phase passes streaming,
concurrency, host restore, alias switching, cancellation and disk reload.
Pooled measured medians, in seconds:

| Input / concurrency | Previous request | Normal request | Previous first output | Normal first output |
|---|---:|---:|---:|---:|
| 16 / 1 | 0.713139 | 0.719510 | 0.059331 | 0.060384 |
| 33 / 1 | 0.813376 | 0.804745 | 0.156251 | 0.138390 |
| 48 / 1 | 0.832196 | 0.807020 | 0.171988 | 0.140711 |
| 64 / 1 | 0.848975 | 0.816180 | 0.187469 | 0.148745 |
| 129 / 1 | 1.040253 | 1.045706 | 0.377131 | 0.378427 |
| 400 / 1 | 1.324421 | 1.330999 | 0.652783 | 0.656469 |
| 129 / 2 | 1.553727 | 1.555254 | 0.513116 | 0.516042 |
| 129 / 4 | 1.861432 | 1.864578 | 1.070697 | 1.072353 |

At 33/48/64 input tokens, first output improves 11.4%/18.2%/20.7% and total
latency improves 1.06%/3.03%/3.86%. Unchanged workloads vary by less than 1%
in total latency. The control disables only the new tile selector; the
candidate uses ordinary dllm. Unlike the earlier experimental run, 129-token
prompts retain their old dispatch and show no material improvement. These
results qualify the bounded A2 launch policy, not general hardware or
cross-engine performance parity.

### Native Q4 residual launch experiment

`q4-residual-launch-a2.json` compares fresh builds of the existing single-row
two-term Q4 kernel with isolated 2/4/8/16-warp variants and a four-warp variant
that removes the unused activation-sum reduction. All use `nvcc -O3
-std=c++17 -arch=sm_86 -shared -Xcompiler -fPIC`; candidate macros are
`DLLM_Q4_LAUNCH_WARPS` and `DLLM_Q4_SKIP_SUM` in
`scripts/experiments/q4_residual_launch.cu`. Production native code is unchanged.

All 216 matrix/range/bias checks pass, including exact agreement with the
production native output and the independent FP16-coefficient RMS gate. Warm
CUDA graph ABBA timing includes both activation quantization and multiplication.
The four-warp experimental control matches the fresh production build within
0.5%. On the large measured shapes, two warps is 7.5–12.9% slower and sixteen
warps 9.8–18.5% slower. Eight warps is mostly 1.6–2.0% slower, except the
4096-by-14336 projection, which is effectively tied. Removing the unused sum
saves at most 0.1% on the large matrices (1.5% on the tiny 113-output tail case).
No production launch or arithmetic changes follow from these results. These
are synthetic kernel measurements, not model or serving performance claims.

The follow-up `q4-residual-cache-a2.json` tests shared activation staging
(`DLLM_Q4_SHARED_INPUTS=1`), activation-plus-scale staging (`=2`), and streaming
packed-weight loads through L2 (`DLLM_Q4_STREAM_WEIGHTS=1`). All 180 exact-output
and independent RMS checks pass. Shared staging slows the large 4096-wide cases
by 18.6–26.7%, and the 14336-wide projection by 265–286%; it is rejected.
Streaming weights improves that wide projection by 3.7% but slows the other
large shapes by 1.5–2.1%. No broad cache policy is promoted. Four-warp calibration
remains within 0.2% of the production control. The separate builds preserve the
same precision and compiler options as the launch experiment.

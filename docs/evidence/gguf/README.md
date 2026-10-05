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
below includes explicit FP32 rotary-factor vectors. K-quants, IQ formats, split
checkpoints, other architectures, other scaled-RoPE metadata, SentencePiece
tokenizers, vision/audio adapters and arbitrary GGUF tokenizer families are
not implemented. Q/K/V and gate/up fused components must each use compatible
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

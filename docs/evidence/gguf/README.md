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

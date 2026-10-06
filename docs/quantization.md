# Quantization and weight caches

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Choose a storage and arithmetic policy together. Qualification is specific to the model, device, precision, and context; import support alone is not numerical or performance proof.

## Per-model precision

Quantization can be selected per alias at registration:

```sh
dllm add chat /path/to/chat/snapshot --quantization q8_0
dllm add encoder /path/to/minilm/snapshot --quantization none
```

`POST /v1/models/register` accepts the same `quantization` field (`none`,
`q8_0`, or `q4_0`), and `/v1/models/status` reports each alias's policy. Omission uses the
server's `DLLM_QUANTIZATION` default captured at startup. Policies are part of
alias identity and remain stable across eviction/reload; register another alias
for a different precision. Qwen2, Qwen3, Llama and Mamba-1/2 Q8 also run on Metal.
[Metal Q8 evidence and performance limits](evidence/quantization/metal/README.md).
Native Metal Q4/Q8 use [bounded packed short-prompt execution](evidence/quantization/metal/short-prefill/README.md)
for FP32 activations; larger prompts keep temporary per-projection expansion.
A Q8 decoder and an
unquantized encoder can share one server and its GPU/host residency budgets.

## Qwen3 arithmetic

For accurate CUDA Qwen3 arithmetic, set `DLLM_QWEN3_PRECISION=float32`.
This retains compact FP16 matrix storage but uses FP32 operations and KV; it
clears the Qwen3-0.6B long-context numerical gate. The default `standard` path
remains faster for batched decoding but retains that checkpoint's documented
accuracy qualification gap. See the measurements in the architecture report.
The accurate Qwen3 mode uses compensated Tensor Core products for decode batches
and longer prompts:
FP32 activations are split into scaled high/residual FP16 components, with FP32
accumulation. Both tested Qwen3 sizes pass independent FP32-reference gates;
this is qualified numerical approximation, not bitwise FP32 equivalence.

## Experimental packed four-bit weights

`dllm add compact /path/to/snapshot --quantization q4_0` enables native
Q4_0 projection weights with FP16 block scales. Input embeddings and norms keep
their existing precision. `DLLM_QUANTIZATION=q4_0` sets the startup default;
per-model policies can mix Q4, Q8 and unquantized models in the same server.

Independent model gates pass Qwen2.5-0.5B, accurate Qwen3-0.6B and
Llama3.2-1B on A2, along with live model-switching tests. The loader currently
allows dense Qwen2/Qwen3/Llama/Phi-3/Mistral only; acceptance alone is not a
checkpoint qualification. Four-bit quantization is lossy: on our small authored
171-token Qwen2 corpus, next-token agreement with the original weights was
74.3% for Q4 versus 95.3% for Q8. This is not a broad quality evaluation.
Q4 remains opt-in. Metal supports Qwen2.5-0.5B geometry with standard RoPE,
context through 2048 and one to four request slots; other Q4 configurations
remain CUDA-only. See [Metal Q4 qualification](evidence/quantization/metal/q4/README.md). This is on-load conversion from safetensors;
Standalone Qwen GGUF import is available below. GPTQ import remains unsupported.

For the qualified Qwen2.5-0.5B CUDA layer partition (split 12/12, context <=512,
batch <=4), adjacent-nibble body storage and packed prompt processing are now
the defaults. This resolves the three greedy-token mismatches in the expanded
282-check stage trace without changing Q4 coefficients, reference targets or
numerical tolerances. The integrated implementation passes A2/A30 and A2/3090
numerical/HTTP checks, plus changed-weight graph and partial-repack recovery.
It repacks existing device buffers; host checkpoints and the output head retain
their original format/path. This does not change ordinary single-GPU Q4 dispatch.

Set `DLLM_Q4_STAGE_ADJACENT=0` for the legacy stage arithmetic and its conservative
sequential-prefill default. `DLLM_PARTITION_PACKED_PREFILL=0` independently disables
prompt packing. Both distributed workers must agree on both policies; model
status reports the active layout. See
[the numerical investigation and measured tradeoffs](evidence/quantization/cuda/q4-arithmetic/README.md).

## Native AWQ checkpoints

CUDA can load dense Qwen2 and Qwen3 checkpoints using AWQ GEMM four-bit packing, group size 128,
and asymmetric zero points. Use `--quantization none` when registering to preserve
the checkpoint format; on-load Q4/Q8 conversion is rejected. GEMV packing,
module exclusions, other group sizes and other architectures are not qualified.

The pinned Qwen2.5-0.5B-Instruct-AWQ checkpoint passes independent HF SDPA
logit checks and fixed/paged, eager/graph, batched/chunked execution through
2048 context on A2. Normal HTTP streaming, cancellation recovery and model
release/reload also pass. Eager FP16 attention has documented numerical
differences, and this is not a general AWQ quality or performance-parity claim.
See [the exact evidence and limitations](evidence/awq/README.md).

Qwen3-4B-AWQ passes 128 independent FP32-reference checks through 2048 context
with both Triton and native Marlin. Both pass fixed/paged, eager/graph, batched
and chunked layout checks; native server lifecycle and switching to the smaller
Qwen2 AWQ checkpoint also pass. It preserves the existing Qwen3 mixed
precision policy: FP16 matrix/KV storage with FP32 residuals and elementwise
intermediates. Its BF16 dense checkpoint tensors are converted to FP16 for GPU
execution; scales remain FP16. Earlier FP16-reference differences are retained
in the evidence rather than described as passing results.

For the qualified Qwen2.5-0.5B and Qwen3-4B AWQ geometries on NVIDIA A2, an optional native
Marlin backend replaces GPU packing and uses shared, accounted scratch memory.
Build it with `-DDLLM_BUILD_AWQ_MARLIN=ON -DCMAKE_CUDA_ARCHITECTURES=86`, then set
`DLLM_AWQ_MARLIN_LIBRARY` to the absolute path of `libdllm-awq-marlin.so` before
starting dllm. The backend reports `triton+awq-marlin`; other devices, model
geometries, contexts above 2048 and batch capacities above four retain the regular
AWQ path. A missing or incompatible library in the selected scope fails explicitly. The vendored
Marlin device kernels retain their Apache-2.0 license and attribution; there
is no vLLM inference runtime dependency. Unset the variable to use Triton.

## Optional CUDA C++ Q4 decode

A separately built native kernel improves single-request Mistral Q4 decode on
A2. Build for your GPU architecture (86 in this A2 example):

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dllm-q4 -j2
export DLLM_Q4_CUDA_LIBRARY="$PWD/build/libdllm-q4.so"
```

Set the library path before starting the server. The runtime checks its ABI
and uses it only for qualified FP16 single-row shapes; other shapes retain the
Triton path. CUDA graph capture uses the caller's current stream. The measured
Mistral Q4 result is 22.53 tokens/s versus 20.06 with the paired Tensor Core
path alone, still below the matched llama.cpp result of 34.20. There is no
runtime compilation requirement when this option is unset.

An additional experimental integer-dot path is available for the qualified
Mistral-style GGUF topology (Llama metadata, 32 layers, width 4096, intermediate
14336, 32 attention/8 KV heads, vocabulary 32000), Q4_0, mixed precision and
context at most 512. Build `dllm-q4-residual` and set
`DLLM_Q4_RESIDUAL_LIBRARY` to that library before starting dllm. Its per-model
selector only changes one-row decode; other shapes and models fall back.
It preserves packed weights but applies scaling in FP32 instead of rounding
each coefficient to FP16. The isolated candidate reduced single-request HTTP
latency 11–18%; final normal-path serving confirmation is tracked in
[the GGUF evidence](evidence/gguf/README.md). This is an explicit opt-in,
not a new default or a broader precision-parity claim.

GGUF Q4_K/Q5_K/Q6_K have a separate CUDA C++ decode library. It reads the
original packed bytes, including mixed-format fused projections, and preserves
FP16 coefficient rounding before FP32 accumulation:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dllm-kquant -j2
export DLLM_KQUANT_CUDA_LIBRARY="$PWD/build/libdllm-kquant.so"
```

Set this before starting the server. It selects native FP16 decode for batches
of one through four when every segment uses a K-quant format. Other cases retain
the Triton fallback. Prompt processing expands one projection temporarily for
the matrix library; persistent weights remain packed. Admission reserves space
for that largest temporary projection in addition to the usual workspace.
An additional opt-in uses two int8 activation terms for faster single-request
Llama Q4/Q6 decoding:

```sh
cmake --build build --target dllm-kquant-residual -j2
export DLLM_KQUANT_RESIDUAL_LIBRARY="$PWD/build/libdllm-kquant-residual.so"
```

Keep the ordinary K-quant library configured for fallback. The residual option
requires CUDA, the default Llama `mixed` precision, hidden width 2048,
only Q4_K/Q6_K packed formats, and a configured context of at most 512
(2048 for the qualified TinyLlama topology). It selects single-row 2048-wide projections; other shapes use the
ordinary path. Q5-containing models are excluded because their independent
numerical gate remains open. The option changes arithmetic without changing
stored weights. A2 Llama 3.2 1B Q4_K_M tests measured 7–11% lower single-request
HTTP latency than ordinary CUDA, with concurrency two roughly tied. Other
hardware, model sizes and long contexts are not qualified by those results.
Published TinyLlama Q4_K_M passes model/layout and normal serving proof through
2048 context. The residual path lowers single-request latency 5.7–8.5% in the
512-context workload and 2.7–8.1% in the separate 2048-context workload.
Other Llama models retain the 512-context residual-selection limit.
Backend status includes `cuda-kquant-residual` when selected.

For the qualified TinyLlama Q4_K_M checkpoint on CUDA, optionally set
`DLLM_KQUANT_DENSE_CACHE_MIB=2048` to retain 1973 MiB of FP16-expanded projection
copies for multi-row execution. Default is zero. Core weights, KV and workspace
reserves take priority; an insufficient cache budget or free VRAM leaves packed
execution in place. `/v1/models/status` reports `gpu_projection_cache_bytes`.
Normal A2 server checks show about 20% lower four-request latency (131 to 164
aggregate output tokens/s) and 4–9% lower single-request latency, with matching
outputs and tested model swaps/reloads. Packed weights remain authoritative;
this is an explicit memory-for-speed tradeoff, qualified through context 2048.

The pinned Qwen3-1.7B Q4_K_M conversion also supports this cache through context
2048 in its standard Qwen3 precision mode. Use
`DLLM_KQUANT_DENSE_CACHE_MIB=4096`; the selected projections occupy 3281.5 MiB.
Normal A2 serving reduces four-request latency about 22% (84 to 107 aggregate
output tokens/s), with exact output agreement and verified reloads, swaps and
insufficient-budget fallback. This remains an optional additional allocation;
it does not enable caching for all Qwen architectures or contexts.

For Llama 3.2 1B Q4_K_M/Q5_K_M, an independent FP32 mode addresses the observed
long-context FP16 numerical gap. An optional FP32 residual library accelerates
its decode without rounding inputs or output to FP16:

```sh
cmake --build build --target dllm-kquant-residual-f32 -j2
export DLLM_LLAMA_PRECISION=float32
export DLLM_KQUANT_RESIDUAL_F32_LIBRARY="$PWD/build/libdllm-kquant-residual-f32.so"
```

Selection requires CUDA, Llama hidden width 2048, FP32 execution, context at most
2048, and Q4_K/Q6_K or Q5_K/Q6_K layouts. It handles one through four rows at
widths 2048/8192; other projections retain ordinary FP32 execution. Persistent
weights remain packed. The separate ABI prevents mixing FP16 and FP32 buffers.
Backend status includes `cuda-kquant-residual-f32`. Independent model checks
cover 2048 context; paging, batching, swapping and HTTP measurements currently
cover 512. Q4 A2 server measurements show 1.41–1.95x speedup over ordinary packed
FP32. This is a different precision policy from the faster FP16 option above;
FP32 prompt processing remains a performance target.

The same Llama FP32 scope also has optional compensated prompt processing:

```sh
export DLLM_KQUANT_PREFILL_F32=1
```

This selects prompts of at least 128 rows at widths 2048/8192. Packed weights
are decoded directly into scaled high/low FP16 components, then three Tensor
Core products accumulate in FP32. This is a numerically qualified approximation
to the unrounded FP32 coefficients. Short prompts retain ordinary processing.
The option is independent of the residual decode library; enabling both combines
their gains. Normal-server A2 Q4/Q5 tests show an additional 9–16% lower latency
on the measured 129/400-token workloads, with short prompts approximately tied. Backend status adds `compensated-kquant-prefill-f32`. Weights remain
packed while resident; the high/low buffers are temporary per projection.

See the [K-quant qualification and performance evidence](evidence/gguf/README.md).

## Prepared weight cache

Set `DLLM_WEIGHT_CACHE=/path/to/trusted/cache` before starting dllm to reuse
prepared Q4/Q8 weights across processes and cold model activations. This is
currently available for dense Qwen2, Qwen3, Llama, Phi-3 and Mistral. The first
load converts weights and writes a safetensors file; subsequent loads reuse it.
GPU-resident and host-resident switching continue to use their existing paths.

Cache keys include the preparation version, validated configuration, precision,
and source shard paths and file metadata. This is local file identity, **not a
cryptographic checkpoint digest**. Use trusted checkpoint and cache directories.
Missing, truncated or incompatible entries are rebuilt; publication is atomic.
Cache write failures warn and leave inference available. Changing a source during
loading is an error. There is no automatic disk eviction: budget roughly one
extra prepared-weight copy per checkpoint and precision, and remove obsolete
cache files when desired. No files are created unless this option is set.

## Standalone GGUF files

Install `dllm[gguf]` alongside the runtime extra. Local `.gguf` files work with
`run`, `serve --model NAME=PATH`, and `add`, including normal host/GPU residency
and model swapping. Keep conversion policy `none`: existing Q4_0/Q8_0 matrix
values and scales are imported directly into dllm's kernels, without
requantization or another inference engine.

```sh
python -m pip install -e '.[runtime,gguf]'
dllm run /models/qwen.gguf --context 512 --prompt 'Explain binary search.'
dllm add qwen /models/qwen.gguf --quantization none
```

Import supports dense Qwen2/Qwen3 and Llama 3 BPE checkpoints, little-endian
GGUF v2/v3, F16/F32, Q4_0/Q8_0 and experimental Q4_K/Q5_K/Q6_K.
K-quants remain experimental. Optional CUDA C++ decode significantly improves
A2 performance; the default Triton path retains a measured gap. See the
qualification report before choosing them for serving.
Floating tensors follow the existing decoder's FP16 weight policy;
this does not provide FP32-weight inference. On CUDA, quantized embeddings stay
packed in host/GPU memory and only selected token rows are dequantized, preserving
the previous FP16 lookup values. The Metal fallback still expands Q8 embeddings
to FP16. Packed output heads stay packed. CUDA supports mixed matrix formats
within Q/K/V and gate/up groups without requantization.
Llama Q/K rotary rows are restored to the decoder layout, including
packed scales, and explicit GGUF rotary-factor vectors remain FP32. Unsupported
architectures, arbitrary SentencePiece variants, split files, other scaled-RoPE metadata,
other quantized formats, or malformed tensors fail explicitly. Q4, K-quants,
and mixed-format fused groups require CUDA.

For a Hugging Face GGUF repository, `pull`, `run`, and `add` accept
`--filename EXACT_FILE.gguf` and optional `--revision`. Only that file is fetched
at an immutable resolved revision; other quantization variants are not downloaded.
Resolved HF cache symlinks work even when their backing blob has no extension.

Llama-style identity-normalized SentencePiece BPE is also accepted for the
`llama` GGUF architecture. Its embedded vocabulary, scores, byte fallback and
chat template need no HF sidecar files. Mistral's pinned original tokenizer
passes native llama.cpp encoding/decoding checks. A vocabulary-corrected Mistral
7B Q4 comparator passes independent full-logit and runtime-layout checks on A2
at 512 context. Explicit uniform `llama.attention.sliding_window` metadata now
selects the Mistral windowed decoder. A separate fixture with the pinned
4096-token window passes independent and fixed/paged/graph/chunked checks
through context 4104, including prompts immediately around the window boundary.
Normal serving also passes exact decoded-trace checks, streaming, concurrency,
cancellation, host restore and disk reload at that context.
Mixed local/global window patterns remain unsupported; KV storage still reserves
context capacity. GGUF and HF fast tokenizers can differ around literal special
tokens, so cross-source comparisons require identical token inputs.
Tied merge scores use native C++ SentencePiece, reconstructed
from the embedded vocabulary without an HF sidecar. The published TinyLlama
1.1B Chat Q4_K_M passes independent tokenizer and model/layout checks through
2048 context, including server lifecycle checks and cross-model swapping. It
matches llama.cpp output in the 512-context workload while remaining 26–79%
slower there. The 2048-context comparison retains long-prompt and concurrent
output differences; independent HF checks agree with dllm on those exact prompts.
Broader qualification remains in issue #46. Structured output on the
positional tokenizer, unused tokens on that path, Unigram and custom
normalization are rejected explicitly.

The embedded Qwen2 BPE tokenizer and chat template need no HF sidecar files.
GGUF tokenization follows llama.cpp's Qwen2 behavior, including preserving
combining marks; some HF tokenizers normalize those inputs to NFC and produce
different token IDs. Import acceptance is not broad model qualification. See
[GGUF validation and limitations](evidence/gguf/README.md).

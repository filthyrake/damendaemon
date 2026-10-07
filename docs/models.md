# Model-specific guidance

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Consult the [support matrix](support.md) and [architecture qualification](architecture-parity.md) for exact tested revisions and remaining gaps. These notes describe architecture-specific usage.

## Sentence embeddings

Sentence embeddings are available through `POST /v1/embeddings`. Register the
pinned `sentence-transformers/all-MiniLM-L6-v2` snapshot at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` like any other model, then send:

```json
{"model":"minilm","input":["A dog plays outside.","A puppy runs in the park."],"encoding_format":"float"}
```

The independent BERT executor supports masked mean pooling and normalized
384-dimensional vectors on CUDA and Apple Silicon. Inputs can be strings or
explicit token-ID sequences, with up to 64 inputs per request. Float and
little-endian FP32 base64 output are supported. `dimensions`, if supplied, must
be 384. Inputs exceeding 256 tokens (including special tokens) are rejected,
not silently truncated. This is qualification of the pinned MiniLM checkpoint,
not all SentenceTransformers models. Encoder quantization is unsupported; register this encoder with
`--quantization none` when the server default is Q8. Requests share
the model-residency scheduler with chat and completion workloads. See
[encoder evidence](architecture-parity.md#sentence-embeddings) for precision,
performance, and current batching limits.

## GPT-2

GPT-2 learned-position decoding is qualified on A2 and Apple M4 with FP32
weights/arithmetic/KV through its complete 1024-token position table. Use the
pinned `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` with `--raw` or `/v1/completions`.
This base model has no chat template. GPT-2 quantization is not yet supported;
register it with `--quantization none` when the server default is Q8.

## Static YaRN configurations

Dense Qwen2/Qwen3/Llama/Mistral checkpoints may use static `rope_scaling` with
`rope_type: yarn`, `factor`, and optional `original_max_position_embeddings`,
`beta_fast`, `beta_slow` and `attention_factor`. On CUDA this selects FP32
arithmetic/KV with FP16 stored matrices to meet the numerical gate, so budget
for the larger KV cache. MPS retains its FP32 execution policy. Full and chunked
model execution is verified on A2/M4 using an explicit Qwen2 configuration
fixture; this does not establish trained 128K-context quality. Dynamic NTK and
LongRoPE remain unsupported.

## GPT-NeoX base models

The independent GPT-NeoX decoder supports partial rotary embeddings and parallel
residual blocks. Pinned Pythia-160M is qualified on A2 and Apple M4 through 2048 context
and the raw completions API. It uses FP32
storage/arithmetic/KV. Use `dllm run PATH --raw --prompt 'Once upon a time'` or
`/v1/completions`; the base checkpoint has no chat template. Quantization,
chunked prefill and prefix reuse are not yet supported for this architecture.

## BLOOM / ALiBi

Pinned BLOOM-560m is qualified through 2048 context on A2 and Apple M4. Its
independent decoder implements ALiBi attention, embedding LayerNorm, head-major
packed QKV and the tied output projection. It uses FP32 weights/arithmetic/KV;
CUDA decode applies ALiBi directly inside the slotted attention kernel. Prefill
bounds additive-mask workspace by processing 128 query positions at a time.
This is internal attention tiling, not scheduler-level chunked prefill.

Use raw completions for the base checkpoint, which has no chat template.
Quantization, prompt-prefix reuse and interleaved prefill are not yet supported
for BLOOM. Acceptance of another BLOOM configuration does not qualify its
checkpoint or an extended context. See [architecture evidence](architecture-parity.md).

## T5 text encoder-decoder

Pinned FLAN-T5-small and original T5-small run independently on CUDA and Apple Silicon, with separate
encoder and decoder context lengths, native GPU projections, and CUDA graph
replay. Use raw completions and `--context 512`, for example:

```sh
dllm run google/flan-t5-small --raw --context 512 --max-tokens 32 \
  --prompt 'translate English to German: How old are you?'
```

Qualification includes full 512-position decoder traces and real streaming,
concurrency, cancellation and model-switching proofs. Initial storage/compute is
FP32. T5 quantization, chat templates, paged KV, prefix snapshots and chunked
prefill are not supported. [Evidence and measured performance](evidence/t5/README.md).

For repeated T5 prompt lengths on CUDA, `DLLM_T5_PREFILL_GRAPHS=1..8` enables
bounded prompt graph caching. It recomputes each prompt with its current tokens.
Warm latency improves, but a new shape incurs capture overhead; default 0 keeps
it off. `--no-graph` also disables prompt graphs. See the T5 evidence for cold
cost, memory bounds and separate core/HTTP measurements.

## Mamba state-space models

Pinned `state-spaces/mamba-130m-hf` and `mamba-370m-hf` run independently on A2 and Apple M4,
with native GPU causal convolution and recurrent-state kernels. Use raw
completions with context up to 2048. Both devices pass independent logits/state
checks and real streaming, concurrent, cancellation and model-switching tests.
Both devices support FP32 and native Q8 matrix weights with FP32
embeddings and recurrent state. Optional chunked prefill interleaves long prompts
with decoding; prefix caching restores complete recurrent snapshots (it cannot
rewind to shorter prefixes). [Latency and correctness evidence](evidence/mamba/chunks/README.md).
Mamba-2 is separately qualified below; Falcon hybrids remain unsupported.
[Matched benchmarks and qualification](evidence/mamba/README.md) include
remaining performance gaps, especially on Apple.

## Mamba-2

The pinned `state-spaces/mamba2-130m` checkpoint runs independently on CUDA and
Metal after offline safetensors preparation. This adds head-wise recurrence and
gated normalization, with FP32 state and FP32 or Q8 matrix weights, raw completions,
chunking, prefix snapshots
and model switching through context 2048. Single-group models only; Codestral
and Falcon hybrids are not covered. [Preparation and real GPU evidence](evidence/mamba2/README.md).

## Direct token-choice scores

For unquantized Qwen2/Qwen3, `POST /v1/token-scores` takes a raw `prompt` and
2–256 distinct `candidate_token_ids`. It returns logits, probabilities normalized
over those candidates, and the selected token without generating an answer.

```json
{"model":"qwen2","prompt":[40,298,1111,13],"candidate_token_ids":[0,1,100,1000]}
```

Choose token IDs appropriate to your model and prompt; the IDs above only show
the request format. Probabilities are **not calibrated confidence**. This is
single-token choice scoring, not multi-token answer scoring or Jev model support.
[Real A2/M4 correctness and latency evidence](evidence/scoring/README.md).



## Trained document reranking

The pinned MS MARCO MiniLM-L6-v2 BERT and multilingual BGE reranker v2 M3
XLM-RoBERTa cross-encoders run independently on CUDA and Metal. Register one as
an alias and call `POST /v1/rerank`:

```json
{
  "model": "reranker",
  "query": "How many people live in Berlin?",
  "documents": ["Berlin had 3.5 million residents.", "Paris is in France."],
  "top_n": 2,
  "return_documents": true
}
```

The response gives original document indices and raw relevance scores, not
confidence probabilities. Up to64 documents are accepted; each query/document
pair must fit the configured context and model limit (512 for MiniLM;
BGE is reference-tested through2048). Oversized inputs
are rejected rather than silently truncated.
[A2/M4 reference, HTTP and performance evidence](evidence/reranking/README.md).
[Multilingual BGE qualification and limits](evidence/reranking/xlm-roberta/README.md).

Mixed-length reranking automatically avoids excessive padding when length
partitioning halves the work. Original document indices and usage are preserved;
[measured A2/M4 gains and limits](evidence/reranking/README.md#mixed-length-padding-reduction).

## Shared MoE routing foundations

The shared `dllm.expert_selection.select_experts` primitive also provides
grouped softmax/sigmoid and square-root-softplus routing foundations for future
modern MoE integration.
It ranks contiguous groups by maximum score, or by the sum of the two highest
scores when a correction bias is supplied. Correction bias changes selection
only; mixture weights come from the original scores, optionally normalized and
scaled. Grouped/biased ties prefer lower group/expert IDs. Decisions remain
device tensors; the existing OLMoE path reuses its unchanged plain softmax/top-k
ordering. Finite inputs representable in FP32 are required; non-softmax underflow
to all-zero selected scores retains zero weights rather than producing NaNs.

Independent CPU scalar-oracle tests cover routing semantics, reordered rows,
ties, masking of negative adjusted scores and normalization. This is not a new
GLM/DeepSeek model loader, shared-expert executor, kernel or GPU qualification.
Reference semantics were checked against the read-only vLLM checkout at
`84bcbc62644356270aaaa5e2d0237d03adc9bb3a`, file
`vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
(Apache-2.0); dllm does not import that runtime.

The pinned [DeepSeek V4 Flash configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)
uses square-root-softplus scores, 256 routed experts, top-6, scale 1.5 and three
initial token-ID-routed layers. Its [official gate reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py)
(MIT) selects those layers' IDs from a token-to-expert table while retaining
weights from unbiased scores. `validate_token_expert_table` checks the checkpoint
table on CPU before device placement; `select_token_experts` then uses device
indexing without host routing decisions and preserves the table's column order.
The caller must validate vocabulary bounds before execution. These are isolated
routing components; compression, hyper-connections, FP4 experts, FP8 scales,
full-model integration and serving are still separate work. Defensive
zero-underflow handling is not a claim of equivalence to the reference's NaN cell.

`dllm.expert_ffn.expert_feed_forward` adds a floating shared/routed expert block.
It accepts already resident, consistently typed FP16/BF16/FP32 projection weights;
it never promotes entire weight matrices implicitly. Gate/up projections feed
FP32 SiLU activation math. Optional clamping is upper-only on the gate and
symmetric on the up projection. Optional FP32 route coefficients are applied
before casting the activated rows for the down projection. Omitting coefficients
computes an independent shared-expert contribution. CPU scalar and low-precision
tests cover this ordering, including a control that distinguishes weighting
before the cast from weighting after the down projection. This is a reusable
block, not routed expert placement, packed FP4/FP8 execution or full-model serving.

### DeepSeek V4 residual-stream components

`dllm.hyper_connections` independently implements the pinned MIT reference's
learned reduction of residual streams, post-layer recombination, and final head
reduction. Mixing and accumulation use FP32 while residual/output storage retains
FP16, BF16 or FP32. The mixing matrix is indexed by source and destination; batched
matrix multiplication avoids a stream-by-stream-by-hidden broadcast temporary.
Sinkhorn normalization preserves the reference's initial row softmax, epsilon
placement, initial column normalization and subsequent row/column iterations.
CPU scalar-oracle tests cover these equations, asymmetric matrix orientation,
reordered and empty batches, dtype preservation and invalid contracts. Hot-path
operations do not extract tensor values to the host. Inputs and learned parameters
must be finite. These primitives do not admit DeepSeek to serving or qualify its
attention compression, packed weights, long context or GPU behavior.

`dllm.packed_tiles.decode_tile` provides CPU diagnostics for raw checkpoint bytes:
E2M1 expert pairs use row/32-column E8M0 scales; E4M3FN uses 128x128-block E8M0
scales. Callers specify a nonempty logical row/column range, with a hard limit of
65,536 decoded elements. Only that tile expands to FP32; inputs can be strided
raw uint8 views. No full projection conversion is implicit. Tests exhaust every
E2M1/E4M3FN code and E8M0 scale, including signed zero, NaN, subnormal values and
FP32 overflow, and cross scale boundaries with unaligned tiles. Format semantics
follow [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf);
scale geometry follows the pinned DeepSeek reference above. This does not replace
packed GPU projections or establish checkpoint numerical qualification.

`dllm.selected_attention.selected_attention` computes the shared-key/value,
per-query selection and learned sink equations using FP32 accumulation. Query
and selected-key chunks bound temporary gathers and scores; the full KV cache is
never cast to FP32. The sink contributes only to the denominator. Negative IDs
are padding, repeated selections count repeatedly, and selection/causal rules
belong to the caller. Empty selections return zero. A scalar oracle covers
chunking, padding, reordered batches, sink effects and large logits. The tensor
component preserves query storage dtype but does not reproduce the reference
accelerator's intermediate probability rounding. Cache compression, selection
construction, model integration and GPU qualification remain separate gates.

`dllm.cache_indices` builds causal selections for full-prefill storage and
single-token ring-cache decode. Logical token positions remain distinct from
physical ring slots. Compressed blocks become visible only when their complete
input block is available. Prefill query slicing preserves the full sequence's
compressed-cache offset; plans exceeding 1,048,576 index elements fail before
allocation. Ring plans support reordered slots with different positions but do
not support simultaneous multi-token writes that could overwrite earlier query
context. CPU integration tests compare ring attention against full-prefill
attention while uninitialized cache entries contain NaNs. The caller still owns
cache writes, compression and position commit/recovery.

`dllm.floating_moe.floating_moe` composes selected floating expert banks and an
optional independent shared expert. Route chunks bound projection gathers and
activation intermediates under an explicit workspace budget; resident weights,
output tensors and backend matmul scratch are separate. Weights retain their
FP16/BF16/FP32 storage. Routes stay as device tensors; no routing counts are copied
to Python. Routed activation weighting precedes the down-projection cast, while
the shared contribution stays unscaled. CPU fixtures cover token-table routing,
reordering, empty input, low-precision composition and chunk budgets. This is not
packed execution, distributed expert placement or full-model qualification.

`dllm.compressed_pool.pool_projected` incrementally pools projected FP32 values
and gate scores with learned positional offsets. Ratio four combines the previous
block's first channel half with the current block's second half; other ratios
use disjoint blocks. Partial blocks survive arbitrary chunking. The function
returns completed vectors, their logical block-start positions and replacement
state; it never mutates published input state. The caller commits replacement
state after downstream cache writes succeed. Scalar tests cover overlap math,
chunk invariance, reordered/shrinking slots and injected failure. Projection
loading, normalization, RoPE, QAT and model integration remain separate gates.

`dllm.compressed_indexer.select_compressed` scores compressed blocks with
weighted ReLU query/key products and selects causal top-k IDs. Query and key
chunks bound score tensors; deterministic equal-score ties choose lower block
IDs. Negative learned head weights remain valid. Completion masks prevent future
or incomplete blocks from entering the result, including poisoned future cache
entries. CPU scalar tests cover scores, IDs, block boundaries, chunk invariance,
reordered inputs and empty caches. FP32 scoring is a component reference;
projection/Hadamard/QAT integration and accelerator rounding remain unqualified.

`dllm.floating_compressor.compress_hidden` connects explicit FP32 projection
weights to pooling, storage-dtype rounding, learned RMS normalization and
adjacent-pair RoPE at completed blocks' start positions. The DeepSeek frequency
schedule is independent of other architectures' YaRN magnitude scaling. Scalar
frequency/rotation tests and full-vs-chunked compressor tests cover the equations;
low-precision controls distinguish the required cast order from late casting.
A downstream rotation failure leaves prior state unchanged. The default result
is an unquantized floating reference. Explicit `cache_transform="kv_fp8"` applies
power-of-two block FP8 simulation only to non-rotary channels;
`cache_transform="index_fp4"` applies normalized Hadamard rotation and 32-value
E2M1 simulation. Scalar rounding and transform-matrix controls cover these modes,
including signed zero, tie-to-even rounding and chunks that complete no block.
Accelerator rounding and real-checkpoint cache behavior remain unqualified.

`dllm.deepseek_checkpoint` describes main-model checkpoint tensor names, physical
packed shapes and expected storage families without materializing values. The
pinned official tensor index matches all 67,612 generated main-model names;
1,575 MTP entries are explicitly reported as excluded and unqualified. A checked-in
configuration/name fingerprint preserves this independent metadata comparison.
The official 44-entry compression schedule includes one MTP layer beyond the 43
main layers; the contract does not misinterpret it as an extra main layer. CPU
fixtures reject missing/extra tensors and wrong packed/scale shapes. Actual shard
headers, tensor contents and full-model execution have not been qualified by this
name-only comparison. The metadata fixture retains its source revision and MIT
license attribution; no checkpoint weight payload was downloaded for the audit.

Run the offline audit with `python -m scripts.audit_deepseek_checkpoint SNAPSHOT`.
It checks config plus shard headers without reading tensor values. Add
`--names-only` when only `config.json` and `model.safetensors.index.json` are
available; this mode does not open or require shards and reports the narrower
scope. Neither mode admits the architecture to serving.

`dllm.floating_indexer.index_hidden` connects floating low-rank query projection,
rotary suffix positioning, Hadamard/FP4 simulation and bounded causal selection
against prepared indexer caches. Head-weight scaling occurs in the projection's
storage dtype before FP32 scoring; the selector explicitly avoids applying that
scale twice. Low-precision controls distinguish this boundary from late scaling.
Projection rows obey the query chunk limit, and reordered/empty input and cache
fixtures pass. Packed FP8 projection arithmetic and full-model/GPU behavior remain
unqualified.

`dllm.floating_attention.attention_prefill` composes complete floating attention
prefill for ordinary windows, ratio-four indexed compression and ratio-128
compression. It connects prepared projections, normalization, cache transforms,
causal selection, sink attention, inverse RoPE and grouped output projections.
Two chunked passes avoid retaining all multi-head queries. Returned cache storage
owns the ring window plus explicitly reserved compressed capacity, rather than a
view retaining the full raw prefill. Pool buffers count toward the cache budget;
weights, outputs and transient workspace are separate. `max_tokens` defaults to
the prefill length; reserve a larger value explicitly for subsequent decode.
CPU dense attention controls and independent grouped projections check composition.

`attention_decode` advances one token per slot in place, using an explicit expected
position and serialized state ownership. Cache history is not concatenated or
reallocated per step. Failed updates after mutation poison the cache; preflight
failures preserve it. Slot reorder/shrink copies are published only after all
copies succeed, within the cache budget; `close()` releases the state's references.
All slots in a state share one position. Tests compare every decode step with a
fresh prefill across ring/compression boundaries and verify failure/release paths.
Prepared weights/frequencies must stay immutable for the cache lifetime. Packed
projection arithmetic, full-model scheduling, GPU synchronization/lifecycle and
performance qualification remain separate; this path is not serving admission.

`dllm.floating_block.FloatingBlock` prepares immutable floating parameters and
composes hyper-connections around attention and shared/routed MoE. Hash tables
are validated once during preparation; token lookups occur before cache mutation.
Prefill/decode tests cover both hash and corrected-score routing, low-precision
storage, reordered slots and independent composition. Cache states retain their
block owner. A downstream FFN failure after attention advancement poisons the
block cache; failed prefill closes its newly created attention state. Parameter
banks are already resident floating tensors, not a packed-checkpoint loader.
Full model/serving and GPU qualification remain open.

`dllm.floating_model.FloatingModel` connects prepared blocks to token embeddings
and the final hyper-connection reduction, learned RMS normalization and FP32
vocabulary head. Prefill returns last-token logits and an owned model cache;
single-token decode requires an explicit expected position. Aggregate cache
budgets cover every layer and constrain slot fanout. A partial layer advance,
head failure or partially published slot reorder poisons the entire model cache.
Failed prefill cleanup remains owned and is retried before another build.
Tiny CPU fixtures compare teacher-forced and greedy cached steps with fresh
prefill, and exercise budget, owner, ordering and failure boundaries. This is a
prepared-floating reference stack: packed checkpoint loading, real-model logit
equivalence, serving admission and GPU qualification remain open.

`dllm.packed_projection.packed_linear` supplies a CPU FP32 projection control
for raw E2M1/E4M3FN weights with E8M0 scales. It expands only one bounded weight
tile at a time and chunks input tokens within a declared conservative tensor
scratch allowance. Resident inputs, the final output, allocator overhead and
BLAS scratch are outside that allowance. Tests compare against independently
decoded scalar weights across scale boundaries, strided and low-precision
inputs, and verify tile release between expansions. It is not a fused GPU
kernel and does not reproduce accelerator activation-quantization rounding.

`dllm.packed_moe.PackedExpert` prepares gate/up/down packed matrices with an
explicit per-expert scratch allowance. `packed_moe` composes selected experts
and an optional unweighted shared expert, preserving coefficient-before-cast
ordering and additive duplicate routes. This CPU control reads route IDs on the
host and processes one token/expert at a time. Tests cover mixed FP4/FP8 storage,
scalar-decoded dense controls, reordered/empty batches, rejected invalid routes
before projection and unchanged packed storage. It is not the device-resident
GPU routing implementation or a real-checkpoint qualification.

`FloatingBlock(..., packed_experts=..., packed_shared=...)` explicitly selects
that packed expert control inside the existing model stack. This mode rejects
floating expert-bank parameters, non-CPU placement, mismatched expert counts,
widths, activation limits and scratch allowances. Attention remains floating.
Two-layer model fixtures with both routing modes compare prefill and cached
decode against independently decoded floating expert controls, exercise slot
fanout and poison the complete cache after a packed expert failure. Full packed
attention, checkpoint loading, accelerator execution and serving remain open.

## GPT-OSS (CUDA)

The pinned 20B checkpoint supports text Harmony chat, auto function calls, and raw completions, with
independent layer/logit, format and HTTP lifecycle qualification on A2. Prepare
the exact HF-format shards with `scripts/prepare_gpt_oss.py`. Its prepared host
weights occupy 12.83 GiB; use a suitable host budget, for example:

```sh
dllm run models/gpt-oss-20b --context 512 --host-gib 18 \
  --reasoning-effort low --prompt 'What is 2 + 2? Answer briefly.'
```

`/v1/chat/completions` accepts `reasoning_effort` (`low`, `medium`, `high`);
the checkpoint defaults to medium. Final answer text is returned as `content`,
analysis separately as `reasoning_content`, including SSE deltas. CLI output
prints the final answer. Assistant responses can be reused as message history;
the checkpoint template discards prior analysis. Token limits count all generated
protocol and analysis tokens, so a small limit can end before answer text begins.
Custom stop strings apply to answer/commentary text, not analysis or headers.

Keep conversion policy `none`: the checkpoint already contains native MXFP4,
reported accurately in response timing metadata. Grouped expert prefill is the
default for prompts of at least 16 tokens. Qualified text-chat scope is A2,
512 context and two slots, plus 1024-context auto-function round trips. Function
calls support `tool_choice="auto"` or `"none"`; complete declared calls appear as
`tool_calls` with JSON arguments, and tool-result history is accepted. A call
truncated by the token limit is not published. Forced/named/strict calls, multiple
calls in one historical assistant message, and tool history containing both
commentary content and reasoning are not yet supported. Structured chat output,
chat logprobs, 120B and longer contexts remain unqualified; unsupported combinations fail
explicitly. Longer prefills use bounded fused attention and IEEE FP32 attention
projections. These repair the tested routing-sensitive output mismatch; 4K HTTP
and tool lifecycles pass, but long-context internal-state numerical qualification
remains open in [issue #41](https://github.com/filthyrake/dllm/issues/41).
See [fused-prefill evidence and scope](evidence/gpt-oss/fused-prefill/README.md). See [architecture evidence](architecture-parity.md#gpt-oss-20b-packed-raw-execution).

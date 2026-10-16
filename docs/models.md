# Model-specific guidance

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Consult the [support matrix](support.md) and [architecture qualification](architecture-parity.md) for exact tested revisions and remaining gaps. These notes describe architecture-specific usage.

Model configuration preflight reads at most 1 MiB of UTF-8 JSON per object
configuration file. The supported Mamba-2 `time_step_limit: [0, Infinity]`
convention is accepted; other nonfinite numbers, duplicate fields and non-object roots
are rejected before architecture dispatch or weight planning. Tokenizer vocabulary
files use their separate loading contracts.

Explicit independent GGUF benchmark overrides require 1..64 nonempty token
prompts and 1..256 output tokens within the requested context. Empty or malformed
reports are rejected before loading the source model, without substituting defaults.

GGUF benchmark/reference/teacher evidence reads are limited to 64 MiB of UTF-8
JSON and reject duplicate fields at every depth. Numerical field policies remain
schema-specific, including diagnostic use of failed teacher traces.

Independent GGUF token-trace reports include `workload_sha256`, binding the
ordered prompt token IDs and output budget. Generated-HTTP qualification requires
that digest to match the requested benchmark workload before starting its server.
Older reference reports without the digest must be regenerated; do not add a
computed digest to historical evidence to imply that workload was validated.
Generated HTTP verification validates complete eager/graph reference cases,
output-token vocabulary bounds, empty mismatch lists, and comparator text before
launching its server. Malformed trace evidence cannot trigger model startup.
HTTP expected text stops at the first checkpoint EOS token, even when the complete
numerical reference trace continues beyond EOS. Both HTTP modes must report exact
prompt, completion and total token counts; streaming must supply one usage record.

Teacher traces require the same workload identity and complete eager/graph case
coverage with valid token IDs before HF construction. A structurally valid failed
numerical trace remains usable for diagnostics; its use does not qualify a model.

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
widths, activation limits and scratch allowances.
Two-layer model fixtures with both routing modes compare prefill and cached
decode against independently decoded floating expert controls, exercise slot
fanout and poison the complete cache after a packed expert failure.

`dllm.prepared_projection.PackedProjection` also connects packed CPU weight
storage to attention query/KV/output and indexer projections. Grouped output
row views retain original matrix coordinates so E8M0 scale blocks stay aligned.
Compressors, norms, routing gates, embeddings and the vocabulary head remain
explicit floating parameters. Ordinary, ratio-four and ratio-128 attention,
cached decode and complete mixed-storage model fixtures match scalar-decoded
dense projection controls. No dense packed-weight expansion is retained.
Scratch allowances exclude projection output buffers and backend scratch;
accelerator activation quantization and accumulation are not emulated.
Whole-model checkpoint loading, GPU qualification and serving admission remain open.

`dllm.packed_checkpoint.read_packed_matrix` reads one packed matrix and its
E8M0 scales from the same or separate local safetensors shards. It validates
bounded headers, exact payload coverage, storage/scale geometry and the combined
raw byte budget before reading either selected payload. Returned tensors own
their raw byte buffers; descriptors close on success or failure. This avoids
depending on typed E8M0 support in the installed safetensors reader. Synthetic
file tests cover both formats, cross-shard reads, malformed/overlapping/truncated
headers, budget rejection and file changes during reading. This is selected
tensor loading, not real-checkpoint or whole-model admission qualification.

The shared checkpoint inventory uses the same bounded, tensor-library-free
header validator when an older safetensors parser rejects an E8M0-containing
shard. The fallback reads metadata only and preserves cross-shard duplicate,
index and payload-range validation. A complete synthetic DeepSeek snapshot
passes the header audit; official checkpoint contents remain unqualified.

`dllm.deepseek_reference_loader.load_reference_model` assembles a local validated
main-model snapshot into that CPU reference stack. It admits all retained raw
or converted parameter bytes plus one largest source tensor before payload reads;
header metadata, projection workspace/outputs and KV caches have separate limits.
It keeps attention/expert matrices packed, converts declared floating parameters,
and omits MTP. FP32 reference storage is the default; BF16/FP16 are explicit
choices. Synthetic checkpoint tests exercise actual loaded model prefill/decode,
exact byte-budget boundaries, configuration rejection, changed-file detection and
descriptor cleanup. This does not register HTTP support, load a tokenizer/chat
template, or qualify official model logits, speed, long context or GPU execution.

`dllm.deepseek_text.DeepSeekText` supplies bounded local tokenizer loading and
independent ordinary V4 text-chat encoding. It checks vocabulary/control markers,
adds BOS exactly once, and preserves the pinned chat/thinking history rules.
The complete rendered prompt, including inserted control markers, must fit the
1 MiB UTF-8 cap before tokenizer encoding.
Five literal prompts, including Unicode and retained/dropped historical reasoning,
match exact IDs from the pinned tokenizer (hash recorded in the text contract
fixture). The ordinary text path explicitly rejects tools, internal task roles,
multimodal content and maximum-effort prompting pending their own integration.
It remains separate from HTTP admission and real-model qualification.

`dllm.deepseek_output.DeepSeekOutput` incrementally separates ordinary generated
reasoning and final content. It retains only a possible control-marker suffix,
handles arbitrary text chunk boundaries, requires EOS for normal completion,
and reports explicit truncation separately. Malformed controls, trailing data
and currently unsupported DSML tool output fail and make the parser unusable.
Previously emitted deltas remain partial output when a later chunk fails.

`dllm.deepseek_generation.ReferenceGenerator` connects the prepared CPU model
and text adapter for bounded greedy reference generation. ByteLevel tokens decode
incrementally, including split UTF-8 sequences and ordinary non-special added
tokens, before channel parsing. Byte-alphabet mapping and whole-piece literal
UTF-8 fallback match the CPU tokenizer; special added controls still require
explicit admission. EOS and
length completion release the cache before the final result; closing an abandoned
iterator also releases it. Failed cleanup stays owned and must succeed before
another prefill. Tests cover scripted failure paths and actual synthetic packed
snapshot generation against fresh-prefix logits. This remains a CPU reference,
with no official-model quality, speed or HTTP support claim.

For a deliberately small local reference snapshot with tokenizer files:

```sh
python -m scripts.run_deepseek_reference SNAPSHOT --prompt 'Hello' \
  --max-new-tokens 16 --context 128 --weight-mib 256 --cache-mib 64
```

This CPU-only command emits JSON lines, starting with its unqualified reference
scope. It checks prompt capacity and declared tokenizer IDs before loading
weights, applies explicit weight/cache/workspace budgets, and closes generation
on broken output pipes. Use `--messages` for a local ordinary-text conversation
file and `--thinking` for the pinned thinking mode. Default weight admission is
intentionally far below the official model size; this is a development entry
point, not a benchmark or automatic full-model deployment.

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

`dllm.deepseek_tools` provides a bounded, independent V4 DSML tool-call codec
from the same pinned MIT format specification. `render_tool_calls` accepts
OpenAI-shaped function calls with JSON object arguments; `parse_tool_calls`
returns complete calls atomically. It supports Unicode strings and finite JSON
values, up to 64 calls, 256 parameters per call and 1 MiB per block. Names are
1–128 characters without quotes, angle brackets or control characters; reserved
model/DSML markers in values are rejected because the wire format has no escape
for them. This check also applies to markers hidden by JSON escapes in nested
values or keys. Duplicate parameters/JSON keys, nonfinite numbers, malformed syntax,
truncated blocks and trailing text fail closed. Normalized argument JSON must
also fit the 1 MiB UTF-8 limit before a parsed call is published; escaped lone
surrogates and JSON escaping that exceeds this limit are rejected. IDs are application metadata and
are not encoded in this wire format. The codec does not execute tools or validate argument schemas. Reference prompting
is described below; HTTP and model qualification gates remain open.

`DeepSeekOutput` and `TokenStream` accept explicit `allow_tool_calls=True` for
structured completion decoding. Ordinary content/reasoning remains incremental;
only the DSML block is buffered within the 1 MiB completion limit. Complete calls
are emitted atomically by `finish()` after block and EOS validation. Partial tool
openers/blocks fail even for length-truncated completions. The pinned tokenizer's
added DSML marker is admitted only in this opt-in mode. Default text decoding
still rejects tool output. The HTTP API does not enable this path.

The CPU reference command accepts `--tools tools.json`, a local list of OpenAI
function schemas. The canonical pinned tool prompt is preserved with its MIT
license in `docs/licenses/deepseek-MIT.txt`. `--messages` can include historical
assistant `tool_calls` and text `tool` results: call IDs must be distinct within
the turn, every call must have exactly one result, and results are rendered in
call order. Calls must name declared tools. When tools are supplied, historical
reasoning is retained according to the pinned format. No input objects are mutated.

The runner emits complete `tool_calls` and `finish_reason: tool_calls`; it never
executes a tool. The caller assigns call IDs before adding returned calls/results
to later history, and validates arguments against the tool schema before use.
Malformed, undeclared or length-truncated calls fail and release the reference
cache. Schemas, histories and expanded prompt/context limits are checked before
weight loading. This is CPU development coverage, not real-model tool accuracy
or production HTTP qualification.

### GLM-5.3 checkpoint contract (metadata only)

`python -m scripts.audit_glm_checkpoint SNAPSHOT --names-only` checks the pinned
`zai-org/GLM-5.3` layout at `aca966e4e02791568aa6a4ced368624b3d897f42`
using only local config/index files. Omit `--names-only` to check local tensor
headers as well, without materializing tensor values. The pinned index SHA-256
is `e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf`.
It contains 117,060 main tensors and 1,569 explicitly excluded MTP tensors.
The index-name fingerprint is checked in the regression suite.

The independent contract covers latent Q/KV projection dimensions, per-layer
full/shared indexer ownership, dense and routed/shared expert matrices, and
128×128 E4M3 block layouts with FP32 `weight_scale_inv` metadata. Geometry follows
the [official Transformers v5.15.0 GLM implementation](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).
Unsupported geometry, quantization changes, missing/extra tensors and header
shape/storage mismatches fail before payload loading. Shared bounded audit
plumbing also serves the DeepSeek metadata command.

The included config fixture retains the pinned [GLM-5.3 license](licenses/glm53-LICENSE.txt).
Names-only success does not prove actual shard shapes, tensor values, numerical
quality, model execution, context capacity or hardware support. The local header
tests use synthetic tensors; official full-shard and model qualification remain
open under #56.

`dllm.glm_indexer.GlmIndexer` is a CPU FP32 primitive for the full-indexer layers:
query/key/head projections, key LayerNorm with epsilon 1e-6, and prefix
interleaved RoPE. `project_keys` returns rotated keys; `select` consumes resident
keys indexed by absolute token position and returns causal top-k IDs/scores.
Unavailable entries use `-1`/negative infinity. Equal scores choose lower token
IDs deterministically; upstream top-k tie ordering is not promised equivalent.
Shared-indexer layers can reuse these selections; persistent cache ownership is
not implemented by this primitive.

Projection/score arithmetic is tiled, with explicit output-storage and conservative
tensor-scratch budgets (64 MiB each by default). These budgets exclude already
resident caller-owned inputs/weights and framework allocator overhead. Queries
may have different logical positions per batch slot. The GLM dot-product scale
is applied before ReLU, separately from the head-weight scale; DeepSeek selector
defaults retain their prior arithmetic. Dense CPU controls cover projections,
normalization, rotation, selection, chunking and budget rejection. Packed FP8
execution, complete MLA attention/model arithmetic and GPU performance remain
unqualified.

`dllm.glm_attention.project_mla` implements CPU FP32 Q/KV low-rank projections,
RMS normalization and tail interleaved query RoPE, returning queries, query
residuals, normalized KV latents and shared rotary keys. The indexer rotates a
prefix instead; these layouts are intentionally distinct. `sparse_latent_attention`
expands only selected KV tiles and combines softmax statistics incrementally,
avoiding a full-context expanded KV tensor or full query/context score matrix.
It accepts distinct causal token IDs with `-1` padding and rejects empty/future/
duplicate selections, nonfinite arithmetic and requests exceeding configured
output/tensor-scratch budgets. Dense full-expanded CPU controls cover projection
and tiled attention equations. These are stateless primitives; persistent state,
complete layers/models, packed precision and hardware behavior remain open.

`CachedGlmAttention` composes the FP32 latent/indexer primitives and output
projection with owned latent, rotary-key and (full-indexer only) index-key
buffers. Cache capacity is reserved before allocation; `max_cache_bytes` bounds
retained cache storage. Batches share one explicit logical cursor. Chunk prefill
and decode publish that cursor only after the output projection succeeds; a
failure after a cache write poisons the state. Shared-indexer layers require the
preceding full layer's matching selections. Wrong owners, stale positions and
capacity overruns are rejected before mutation.

Slot reordering/shrinking/duplication prepares all replacement buffers before
publishing them. Its budget bounds retained storage, not the temporary overlap
with old buffers. `close()` releases owned buffers; failed prefill closes its
partially constructed state. CPU tests compare incremental output to fresh-prefix
execution and cover full/shared layers, reorder allocation failures, post-write
failures and cleanup. This remains an FP32 reference attention component, without
packed weights, complete-model qualification or production serving admission.

`GlmFeedForward` implements CPU FP32 dense and routed/shared SiLU FFNs. Sparse
routing uses sigmoid scores, correction bias only for selecting experts,
`sum + 1e-20` normalization and post-down-projection route weights. Token chunks
use views of resident expert matrices rather than gathering whole expert banks.
Output and explicit scratch limits exclude resident parameters and backend
allocator/matmul overhead. Deterministic biased ties prefer lower expert IDs.

`GlmBlock` composes input RMS normalization, cached attention, residual addition,
post-attention RMS normalization and the dense/MoE FFN. Its block cursor commits
after the FFN succeeds; downstream failure after attention mutation poisons both
states. Dense/sparse expert controls, tiny-score routing and cached/fresh-prefix
block comparisons are local FP32 evidence only. Complete-model, actual packed
checkpoint and hardware qualification remain open.

`GlmModel` composes prepared CPU FP32 embedding, mixed dense/sparse blocks,
full/shared indexers, final RMS norm and vocabulary head. Aggregate retained
cache admission precedes every layer allocation. It returns last-token logits
by default; explicit all-token logits have a separate output budget. Cache
cursors publish only after every block and the head succeed. Partial layer
advances or cross-layer reorder failures poison the entire model state.
Failed allocation/prefill cleanup retains ownership for retry before another
allocation. Callers must close returned states when finished.

The tiny prepared-model controls compare incremental logits with full-prefix
and manually composed stack results, and cover reordered slots, aggregate
budgets, cursor drift, head failures and cleanup retry. The CPU primitives use
explicit CPU output/cache allocation. This completes the prepared FP32 reference
stack, not actual GLM FP8 snapshot loading, tokenizer/command integration, trained
model quality or hardware qualification.

The shared packed tile reader, projection adapter and selected checkpoint-pair
loader now accept explicit `e4m3fn_f32`: raw E4M3FN byte weights with FP32
128x128 block scales. Existing `e4m3fn` continues to mean E8M0 byte scales;
the formats are never inferred from values. The raw loading budget includes
four bytes per FP32 scale and rejects mismatched storage before reading values.
FP32 checkpoint reads require a little-endian host. Only selected tiles expand;
row views retain their original block coordinates. Special scale values propagate
for diagnostics. This is a CPU FP32 arithmetic control, without accelerator
activation quantization or actual GLM checkpoint qualification.

GLM latent Q/KV projections, selected KV expansion, full-indexer Q/K
projections and the attention output accept these explicit packed adapters
with FP32 results. Norms and index head weights remain floating. Full/shared
cache tests compare against independently decoded dense weights, including
chunked prefill, duplicated/reordered slots and failures before/after cache
mutation. Nonfinite index keys now fail before cache writes. Each packed
adapter has its own explicit scratch allowance, additional to attention tensor
scratch; these are not a combined process-memory cap. No dense weight matrix
is retained and no accelerator activation rounding is emulated.

`PackedGlmFeedForward` composes prepared E4M3FN/FP32-scale dense or routed/shared
experts with the same GLM blocks/model. Routing and correction bias remain FP32;
coefficients multiply expert outputs after the down projection. One token and
one selected expert execute at a time. Its scratch preflight includes the
largest expert allowance plus routing/accumulation storage; output has a separate
cap. Weights remain packed, and optional DeepSeek activation clipping is rejected.
Mixed dense/sparse, full/shared packed-model tests compare incremental/reordered
logits with independently decoded dense weights and poison all cache layers after
a nonfinite expert failure. These synthetic controls establish prepared CPU
composition, not actual checkpoint loading, trained-model quality or GPU speed.

`glm_reference_loader.load_reference_model` admits local indexed or single-shard
snapshots into that packed CPU stack. It validates configuration/component
limits, complete headers/index agreement, typed storage and projection/expert
workspace before reading values. The default 256 MiB parameter cap includes
retained packed weights/FP32 floating parameters plus one largest source tensor
for conversion overlap; real GLM-5.3 weights exceed this default. At most 256
open shards and 64 MiB aggregate raw headers are admitted. MTP is inventoried
but its values are skipped. Owned descriptors close on failures, and changed
files are rejected. Shared index reading now bounds the read itself, eliminating
a stat-versus-file-growth gap. Synthetic snapshot controls match independent
native FP8 decoding and full/cached logits; official checkpoint values, tokenizer
integration, serving and hardware qualification remain open.

`GlmText` independently renders the pinned ordinary text template, including
default Max reasoning effort, explicit Low/High, assistant history extraction
and optional history-reasoning clearing. The official template always opens
`<think>`; this adapter does not invent a non-thinking switch. Tools, media,
developer messages and incomplete histories fail explicitly. Inputs and local
metadata reads are bounded, and no remote tokenizer code is imported. Six
pinned official-template prompt/token-ID fixtures were checked with the actual
local tokenizer. Its 154,856 contiguous mapped IDs occupy a 154,880-row model
vocabulary; the 24 padded rows cannot be decoded as text. EOS, next-user and
observation markers are the declared generation stops. The tokenizer/template
revision and file hashes are recorded in `tests/fixtures/glm53_text_contract.json`;
the preserved GLM license remains in `docs/licenses/glm53-LICENSE.txt`. This is
text-format evidence, not trained-model generation or tool support.

`glm_generation.ReferenceGenerator` and `python -m scripts.run_glm_reference`
provide explicit ordinary greedy CPU generation. The command validates text,
stop IDs, context and component budgets before loading weights, emits JSON
reasoning/content deltas, and labels output as unqualified CPU-reference work.
Incremental ByteLevel decoding is shared with DeepSeek and preserves split UTF-8
without repeatedly decoding history. GLM padded vocabulary rows are excluded
from selection; unsupported tool/media controls and nonfinite logits fail.
Cache release precedes the completion event and runs on length, stop, iterator
close and errors. Failed releases remain owned for retry before another prefill;
primary decode errors survive cleanup failures. Tests include a real tiny packed
snapshot through the command with a controlled synthetic head. They establish
wiring and ownership, not useful trained-model output or accelerator performance.

Example for an explicitly admitted local fixture:

```sh
python -m scripts.run_glm_reference /path/to/snapshot --prompt 'Hello' \
  --max-new-tokens 16 --context 128 --weight-mib 256 --cache-mib 64
```

The default weight cap deliberately cannot admit the full official GLM-5.3
snapshot. No downloads, model registration or hardware jobs are started by this
command. Full official-weight, tool, HTTP and long-context qualification remain
separate open gates.

Dense and AWQ checkpoint metadata readers bound `tokenizer_config.json` to 16 MiB (including added-token metadata) and `generation_config.json` to 1 MiB. Both require UTF-8 JSON objects with unique fields at every depth; excessive nesting is a validation error. Registration uses the same tokenizer metadata reader before weight inventory or alias publication. These metadata limits do not apply to the vocabulary file `tokenizer.json`.

Checkpoint chat templates receive `bos_token` and `eos_token` as text even when tokenizer metadata stores them as structured AddedToken objects: the `content` string is used. Missing values remain `None`; malformed structured entries are rejected before registering chat-capable models.

When no embedded `chat_template` is supplied, `chat_template.jinja` is read as UTF-8 with the same 16 MiB metadata envelope. An embedded template, including an empty string, takes precedence; a missing fallback preserves the raw-completion-only behavior. Explicit operator overrides retain their separate 128 KiB limit.

Sentence encoder `modules.json` is limited to 1 MiB of UTF-8 JSON and rejects duplicate fields or excessive nesting before pooling configuration or checkpoint loading. The supported pipeline remains Transformer → mean Pooling → Normalize.

Native chat tokenization consumes template output incrementally and rejects more than 16 MiB of rendered UTF-8 text, including a forced tool-call prefix, before tokenization. The template generator closes on success or failure. This bounds accumulated prompt output. Rendering also caps explicit Jinja for-loop work at 1,000,000 iterations per render, shared across nested/recursive loops and nested renders; explicit loop-filter candidates count even when rejected. Independent requests have independent budgets. These limits do not bound intermediate expression/filter allocations or establish a general execution deadline.

The CPU reference chat path retains its 1 MiB rendered-output limit and uses the same text-buffer approach so empty template chunks do not accumulate retained objects. Both paths close their template generators on overflow.

SentencePiece GGUF supports unique-spelling UNUSED pieces in both static-score and positional-score paths: they participate in encoding but emit no decoded text. Serialization and logprob presentation preserve this behavior. Duplicate spellings, ambiguous equal-byte-length overlapping specials, and UNUSED stop IDs are rejected. See [the independent CPU oracle coverage](evidence/gguf/sentencepiece-unused.md); this is tokenizer compatibility evidence, not GPU qualification.

GGUF Qwen2/Llama3 byte-level and SentencePiece tokenizers share longest-special-first partitioning, including overlapping special spellings. Equal-byte-length ambiguous overlaps are rejected. The wrapper policy survives checkpoint-plan serialization; independent CPU fixtures cover both families.

Qwen2/Llama3 byte-level GGUF also retains unique UNUSED pieces for encoding while suppressing their decoded output. Suppression survives serialization and applies to incremental streaming and logprob text/bytes; CPU-qualified structured grammars exclude these invisible pieces from their masks.

Chat-template evaluation failures (including expression type errors, arithmetic errors and unsupported JSON-filter values) are rejected before tokenization/inference as invalid requests. Both native and CPU-reference chat preserve the original exception cause; unrelated runtime failures are not converted into request errors.

Opt-in repetition/presence/frequency penalties and logit bias reject non-finite adjusted FP32 scores before selection. Finite parameters can still overflow that arithmetic. This validity check may synchronize the device when adjustments are enabled; the default unadjusted sampling path is unchanged, and GPU overhead has not been measured.

Structured output uses explicit generated-byte vocabularies for GGUF tokenizers including literal user-defined pieces, UNUSED suppression and positional SentencePiece encoding. CPU tests against pinned xgrammar cover literal Unicode pieces, partial UTF-8 bytes, serialized tokenizers, hidden controls and padded rows. CPU matcher tests cover static and positional SentencePiece with and without a space prefix, original and serialized forms, and UNUSED suppression. This is not GPU qualification.

Experimental Qwen3.5 image loading validates vision config/rotary containers and exact integer geometry before either text or vision weight materialization. Malformed checkpoint metadata fails with a configuration error; CPU preflight tests do not qualify image inference on GPUs.

Image prompt expansion is checked against model context using planned grids before resize, normalization or FP32 patch construction. Inline decoding remains separately byte/pixel bounded; runtime admission still checks the requested output-token reservation.

Direct image executor admission validates placeholder/grid geometry before calling the vision encoder. Mixed batches preflight all image metadata before dispatching their first prompt; direct feature-prefill retains the same checks and exact integer slot bounds. CPU sentinel tests establish this call ordering, not accelerator execution correctness.

Safetensors inventory preflights every header as bounded, duplicate-free UTF-8 JSON (16 MiB maximum) before native parsing. Duplicate tensor names, descriptor fields and metadata keys are rejected in ordinary and E8M0 checkpoints. This preflight reads metadata only; native dtype interpretation remains with safetensors, with the existing explicit E8M0 compatibility fallback.

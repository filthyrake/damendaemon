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

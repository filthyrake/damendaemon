# Independent BERT cross-encoder reranking

Issue #35. Qualified checkpoint:
[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, Apache-2.0.
Host FP32 weights:90,854,404 bytes. This is a trained relevance model, distinct
from token-choice scoring and from Jev/Open-Jev decision models.

The native encoder shares BERT attention/LayerNorm/GELU with sentence embeddings,
but uses paired query/document tokenization, segment embeddings, CLS pooling,
tanh pooler and the checkpoint's trained scalar classifier. FP32 arithmetic and
weights are used on both CUDA and Metal. No Transformers inference is called in
production; Transformers is the independent validation reference only.

`POST /v1/rerank` takes `model`, nonempty `query`,1–64 nonempty text `documents`,
optional `top_n` and boolean `return_documents`. Results contain original indices
and raw `relevance_score`, sorted descending with original order breaking ties.
These scores are not normalized probabilities or calibrated confidence. Usage
counts all encoded query/document pairs, including repeated query/special tokens.
Overlength pairs fail explicitly; there is no silent truncation. The model's
512-token limit is also bounded by the configured service context.

Scope is BERT absolute-position/GELU with one raw scalar sequence-classification
output. Quantization, other pooling/activation conventions, multiple labels,
multimodal inputs and other reranker architectures are not qualified. Rerank,
embedding and generation task mismatches fail with HTTP400. Requests use bounded
scheduler commands and model residency; separate rerank requests are serialized,
while document pairs inside one request execute together.

## Independent reference and lifecycle proof

A2 and M4 each pass17 CPU FP32 Transformers eager reference cases. Worst absolute
score errors:2.86103e-6 A2 and4.29154e-6 M4, below the2e-4 absolute/relative gate.
Coverage includes single and mixed batches, Unicode, padding, pair token/type-ID
identity, changing segment boundaries at equal padded shape, graph churn,
returned-buffer independence,512-token input and513-token rejection. The authored
Berlin example ranks the population passage above unrelated passages; this is a
smoke test, not a retrieval-quality evaluation.

Live HTTP tests on both GPUs pass top_n, returned-document/index mapping, exact
usage, duplicate documents, concurrent request isolation, invalid-input recovery
and reranker/embedding/chat model switching with host restoration. Original
MiniLM sentence embeddings still pass their full independent reference suite on
both GPUs after sharing the implementation.

The mixed A2 workload exposed a PyTorch graph-capture ordering problem after
encoder inference: capture_begin attempted to update inference-mode RNG bookkeeping
before inference mode had been entered. Decoder capture paths now enter inference
mode before graph capture. The formerly failing rerank→embed→chat sequence passes.

CUDA retains at most four encoder graphs for small pair batches (up to8), keyed
by batch size and padded length; each replay updates IDs, masks and segment IDs.
Larger batches and Metal use eager execution. No generated tokens are involved.

## Measured HTTP throughput

Same four short authored passages repeated to fill each batch,10 measured rounds
after one warmup, single client, complete loopback HTTP request. Models resident;
no model loading in warm measurements. FP32, same checkpoint on both devices.

| Pairs/request | A2 median ms | A2 pairs/s | M4 median ms | M4 pairs/s |
|---|---:|---:|---:|---:|
| 1 | 5.193 | 192.6 | 3.587 | 278.8 |
| 4 | 7.100 | 563.3 | 3.638 | 1099.4 |
| 16 | 12.466 | 1283.5 | 7.814 | 2047.7 |
| 64 | 35.468 | 1804.4 | 21.177 | 3022.2 |

These are narrow warm-workload measurements, not general performance parity or
a comparison against vLLM, llama.cpp, the model-card V100 figures, or Jev. Raw
trials, score examples and reference checks are retained alongside this report.

## Matched A2 vLLM comparison

`comparison/` retains all HTTP trials and numerical diagnostics. Same pinned
checkpoint, A2 UUID, FP32 weights/arithmetic,512-token service limit, query and
four repeated short documents. One warmup followed by five measured rounds for
each workload; warm models, loopback HTTP, no model-loading time. Latency is the
median individual request time; throughput is the median aggregate pairs/s per
round. These limited samples are not a general retrieval benchmark.

| Pairs/request | Clients | dllm ms | vLLM ms | dllm pairs/s | vLLM pairs/s |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 5.81 | 18.96 | 162 | 52 |
| 1 | 4 | 12.29 | 27.21 | 246 | 131 |
| 4 | 1 | 6.59 | 29.56 | 584 | 134 |
| 4 | 4 | 14.35 | 44.58 | 758 | 325 |
| 16 | 1 | 11.99 | 43.18 | 1307 | 367 |
| 16 | 4 | 31.43 | 91.30 | 1575 | 599 |
| 64 | 1 | 28.98 | 95.46 | 2189 | 668 |
| 64 | 4 | 82.64 | 231.28 | 2319 | 983 |

All240 responses, including warmups, pass score agreement within3.34e-6 and
identical token usage. dllm uses torch2.6+cu124; baseline vLLM0.30.0 uses
torch2.13.0+cu130, Triton3.7.1 and Transformers5.18.0. Versions differ because
these are independently installed supported engine stacks; this is an engine
comparison, not an isolated kernel comparison. vLLM uses its pooling runner,
TRITON_ATTN, default compilation and supported graphs. Commands are in reports.

Two baseline details matter for reproducibility:

- This checkpoint declares an identity output activation. In installed vLLM0.30,
  `use_activation:false` also disables BERT's required internal pooler tanh,
  giving10.854 instead of8.584. `use_activation:true` preserves the pooler and
  still returns raw relevance. `pooler-diagnostic.json` independently isolates
  the missing tanh; the initial failed baseline is retained.
- Default Triton FP32 dot products use reduced-precision multiplication. The
  corrected-activation default run still differs by about0.0032. The matched
  run sets `TRITON_F32_DEFAULT=ieee`, `VLLM_FLOAT32_MATMUL_PRECISION=highest` and
  `NVIDIA_TF32_OVERRIDE=0`. Default-arithmetic timings remain in the evidence
  but are not silently treated as numerically equivalent FP32 results.

No upstream files were modified. Reproduce with `scripts/compare_rerank_a2.py`
on the authorized A2. It validates every response across both engines.

## Larger-batch graph experiment: not enabled

`experiments/` retains the experimental patch, GPU verification and HTTP trials.
The patch was removed from the runtime after testing. To reproduce, apply it
only in a disposable checkout, set `PYTHONPATH=.` and run its verification script.
It allowed64-pair graphs with a4096-padded-token bound, retaining the four-entry
LRU. Replay with changed IDs and segments was exact; eviction and graph clearing
released retained graph objects. This is experimental evidence, not a supported
runtime environment setting.

GPU-only eager/replay medians were5.204/5.087ms at16 pairs and18.449/18.579ms
at64 pairs, with129/183ms cold capture cost. HTTP medians moved11.995→10.212ms
and28.980→28.370ms, but unchanged one-pair control also moved5.813→5.004ms.
The experiment does not establish a substantial repeatable larger-batch gain.

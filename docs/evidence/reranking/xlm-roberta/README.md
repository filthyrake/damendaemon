# Multilingual XLM-RoBERTa reranking

Issue #36. Independent implementation for
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e),
pinned revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, Apache-2.0.
FP32 host/device weights total2,271,023,108 bytes. Production uses damendaemon's shared
encoder operations; Transformers is only the independent CPU test reference.

The loader accepts the XLMRobertaForSequenceClassification architecture with one
raw scalar output, absolute positions and GELU. It maps the RoBERTa backbone and
trained dense/tanh/output classifier independently. Position IDs count non-pad
tokens from pad_id+1; explicit padding tokens inside text skip positions without
changing the tokenizer's attention-mask semantics. Paired special tokens and
zero segment IDs come from the pinned tokenizer. The classifier returns raw
relevance, not sigmoid probabilities or calibrated confidence.

The checkpoint's position table has8194 entries with pad ID1, allowing8192 input
tokens. GPU reference qualification here stops at2048 tokens, with boundary
cases at512/1024/2048 and explicit2049 rejection for a2048-context executor.
This does not establish full8192-context performance, maximum batch capacity or
support for other XLM-RoBERTa pooling/classification variants.

## Correctness and serving

Each device's independent reference suite covers English, Chinese, French,
German, Japanese, Arabic, mixed Unicode, single/mixed batches, internal literal
padding, repeated graph replay, long inputs, multiple length buckets and reverse
order. English relevant-vs-distractor ordering is a smoke test; multilingual
score agreement is not a retrieval-quality benchmark.

Long-context buckets extend the BERT policy using powers of two with a64-token
minimum. Partitioning still requires4096 padded tokens and at least50% saved
padding. Variable partitions bypass graph capture and restore original indices.

Live HTTP qualification checks original document mapping, top_n, duplicate
scores, concurrent request isolation, exact token usage, wrong-task/overlength
recovery, rerank→embedding→chat→rerank switching and host restoration. The
original BERT reranker is rechecked independently after the shared changes.

## Short-document HTTP measurements

Same four authored Berlin/Paris/puppy passages repeated to fill a request,
FP32,512 service context, one warmup and ten measured rounds, one client, resident
model. Complete loopback HTTP latency includes tokenization, scheduling, GPU
execution and serialization. These are damendaemon measurements, not a cross-engine
comparison or a general multilingual benchmark.

| Pairs/request | A2 median ms | A2 pairs/s | M4 median ms | M4 pairs/s |
|---|---:|---:|---:|---:|
| 1 | 17.620 | 56.75 | 21.737 | 46.00 |
| 4 | 38.833 | 103.01 | 33.409 | 119.73 |
| 16 | 109.165 | 146.57 | 98.322 | 162.73 |
| 64 | 372.373 | 171.87 | 354.087 | 180.75 |

Reproduce the reference with `scripts/verify_xlm_reranker.py`; HTTP/lifecycle
uses `scripts/verify_reranker_service.py` with the BGE model, qualified MiniLM
embedding model and Qwen2 chat model. The checkpoint download contains no custom
execution code. Shared behavior remains FP32-only for reranking, at most64
text documents per request, with no silent truncation and no quantized classifier
or embedding-mode support for this new family.

Both GPUs pass14 numerical cases at the2e-4 absolute/relative reference gate.
Worst absolute error is2.003e-5 on A2 and1.431e-5 on M4. The original BERT
reranker passes its17-case reference suite on each GPU after the shared changes.
Full unit/HTTP regression suite:208 tests pass.

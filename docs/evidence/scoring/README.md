# Direct candidate-token scoring

Issue #33. Independent implementation inspired by the typed-decision pattern and
indexed output projections surveyed in [upstream research](../../upstream-inference-research.md).
No upstream source is copied. This does not load or reproduce Jev model weights.

`POST /v1/token-scores` accepts `model`, raw `prompt` (text or token IDs) and
`candidate_token_ids` (2–256 distinct IDs). It runs eager prompt inference once,
selecting only the requested output-head rows. It returns the selected index/ID,
raw logits and probabilities normalized over that candidate set, `calibrated:false`
and zero completion tokens. This is next-token scoring with explicit verbalizers,
not multi-token sequence likelihood, trained classification or calibrated confidence.

Initial scope: unquantized Qwen2/Qwen3, CUDA and Metal. Requests are bounded
scheduler commands and serialize with generation and model management. Scoring requests are not batched and do not reuse prompt prefixes. The default
eager path keeps no extra persistent head. With optional CUDA prefill graphs,
selected heads live only in the bounded graph cache, keyed by exact candidate
IDs and order as well as prompt length/slot. The temporary executor override is
cleared even when prefill raises; ordinary captured decode retains its full head. Q8/Q4 and other families fail explicitly.

## Qualification and timings

Pinned Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775` and
Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`, context2048, identical
16/256/1024-token raw ID prompts, 2/32/256 candidates. A2 uses damendaemon's Qwen2
standard path and Qwen3 FP32 path; M4 uses native Metal FP32 arithmetic. Both
sides of each comparison use the same resident executor and weights.

All candidate argmaxes agree with the full-head calculation. Maximum candidate
logit discrepancy is 1.90735e-6 on A2 and zero on M4 across this matrix. HTTP
probabilities also agree with the independent full-head gather. These are
mathematical/path checks, not task-quality or calibration evaluation.

Median synchronized eager prefill latency, 15 trials after 3 warmups. Narrow
includes output-weight extraction per call; full includes gathering the same
candidate logits. Both include the complete backbone. Loading and HTTP are
excluded. Full trials precede narrow trials at each point; these measurements
are a first comparison, not a thermally balanced significance study.

| GPU/model | Prompt tokens | Candidates | Full head ms | Narrow head ms |
|---|---:|---:|---:|---:|
| A2 Qwen2 | 16 | 2 | 22.213 | 20.491 |
| A2 Qwen3 | 16 | 2 | 44.763 | 43.667 |
| M4 Qwen2 | 16 | 2 | 30.375 | 25.258 |
| M4 Qwen3 | 16 | 2 | 35.695 | 28.657 |
| A2 Qwen2 | 1024 | 256 | 72.344 | 70.995 |
| A2 Qwen3 | 1024 | 256 | 598.806 | 598.358 |
| M4 Qwen2 | 1024 | 256 | 373.425 | 374.137 |
| M4 Qwen3 | 1024 | 256 | 614.439 | 624.725 |

All intermediate rows, including regressions, are retained in the JSON reports.
Output-head reduction helps short prompts, but backbone cost dominates long
prompts. This does not establish an across-the-board speedup. No vLLM/llama.cpp
or Jev speed comparison is claimed. Avoiding answer decoding is a capability;
a matched end-to-end decision workload remains to be benchmarked.

Live HTTP checks cover distinct candidate sets from concurrent clients, model
switch/host restoration and generation after scoring. Lifecycle checks on M4
and A2 paged KV prove unchanged greedy generation before/after successful and
invalid scoring requests, interleaved score/generate commands, released pages
and no residual head override. Unit/API tests cover bounds, copies, malformed
requests and forced prefill failure cleanup.

## Optional candidate-specific CUDA graphs

`DAMENDAEMON_PREFILL_GRAPHS=1..8` now also covers scoring, with the same default128-token
ceiling. Exact ordered candidate IDs are part of the key; a selected head is
retained with its graph so replay cannot read a freed or different head. A full
generation head uses a separate key. Candidate heads share the existing entry
budget, so choice-set churn can evict ordinary prompt graphs and vice versa.

`verify_scoring_graphs.py` passes on both pinned Qwen checkpoints: changed prompt
IDs, reordered candidates, different same-size choice sets, head eviction/release,
recapture and full-vocabulary generation. Lifecycle interleaving and failure
recovery also pass with graphs enabled. Scores match the full-head reference
within the 5e-5 absolute/relative gate. Graph support remains CUDA-only.

Complete four-choice scoring call, median15 synchronized trials after3 warmups,
including weight-row extraction and token upload, excluding HTTP/loading:

| A2 model | Tokens | Eager ms | Cached graph ms |
|---|---:|---:|---:|
| Qwen2 | 16 | 20.339 | 6.193 |
| Qwen2 | 128 | 21.931 | 9.860 |
| Qwen3 FP32 | 16 | 47.505 | 7.977 |
| Qwen3 FP32 | 128 | 49.268 | 29.350 |

Matched Qwen2 loopback HTTP (`bench_scoring_http.py`), preloaded model, same raw
prompt and four candidate IDs,15 measured requests following one separate first
request. First request includes remaining warmup/capture work, not model loading:

| Tokens | Graphs off warm ms | Graphs on warm ms | Off first ms | On first ms |
|---|---:|---:|---:|---:|
| 16 | 26.218 | 10.983 | 86.762 | 197.739 |
| 128 | 25.691 | 14.472 | 34.882 | 189.495 |

The complete score vectors agree. Trials are sequential off-then-on, single
client, repeated inputs; they are not a broad latency or quality benchmark.
Capture cost, changing choice sets and graph memory can outweigh warm gains.
No Jev, vLLM or llama.cpp comparison is claimed.

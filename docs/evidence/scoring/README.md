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
scheduler commands and serialize with generation and model management. This first
implementation does not batch scoring requests, reuse prompt prefixes or run
scoring in captured graphs. No persistent extra head is cached. The temporary
head is cleared even when prefill raises, and ordinary captured decode retains
its full output head. Q8/Q4 and other families fail explicitly.

## Qualification and timings

Pinned Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775` and
Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`, context2048, identical
16/256/1024-token raw ID prompts, 2/32/256 candidates. A2 uses dllm's Qwen2
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

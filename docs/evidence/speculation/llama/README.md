# Llama numerical gate and explicit precision tradeoff

Issue #37. Pinned `unsloth/Llama-3.2-1B-Instruct` revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`, NVIDIA A2. No change to the
existing default `DLLM_LLAMA_PRECISION=mixed` policy is implied.

## Why mixed precision is excluded

All 18 verifier width/slot/acceptance cases fail exact greedy agreement at output
index 4 under the mixed policy. The ordinary path emits token 1473; the verifier
path emits 315. The other slot's KV remains unchanged in every case. A controlled
probe starts sequential and parallel causal rows from identical KV and sees
shape-dependent rounding in the first matrix projection, before attention.
At the near-tied prediction, sequential logits for these two tokens differ by
only 0.015625. This is numerical divergence, not permission to silently waive the
exact-token gate. Reports preserve both token sequences and per-layer errors.

The explicit `DLLM_LLAMA_PRECISION=float32` experiment retains FP16 stored
matrices, uses FP32 residuals/KV/elementwise arithmetic and the existing dense
FP32 arithmetic path (including compensated Tensor Core products for small
multirow projections). It passes all 18 verifier cases; this is observed
agreement on the qualified inputs, not a mathematical proof of batch invariance.
Mixed/standard, quantized and paged Llama remain excluded from speculation.

## Complete HTTP A/B at the same higher precision

Width8, context512, 64 output tokens, one client, one resident model, six
alternating baseline/optimized rounds after warmup. Both modes use the explicit
higher-precision policy. Full response text, finish reason and usage agree.

| Workload | Ordinary ms | Speculative ms | Speedup |
|---|---:|---:|---:|
| Repetition | 1216.411 | 218.377 | 5.57× |
| Explanation | 1213.700 | 1150.590 | 1.05× |
| Code/list | 1212.516 | 719.013 | 1.69× |

These speedups do not compare against the faster default mixed policy.
Streaming, seeded sampling/penalties/logprobs/stops/schema fallback, slow
consumption, disconnect recovery, four-request concurrency, graph reclamation
and host restoration pass in the same live HTTP suite.

## Long-context memory finding

The first unchunked 8192-context reference replay runs out of A2 memory: the
FP32 GQA math-attention backend attempts a 7.97 GiB temporary allocation. This
failure is separate from the passing context512 serving qualification. The same independent FP32 reference trace passes all82 comparisons with
128-token chunked prefill, including rolling slots and decode through context8192;
maximum absolute logit error is0.000120. Chunked scheduling continues to fall
back to ordinary generation. The unchunked allocation is a separate follow-up.
The full regression suite passes216 tests.

## Actual precision-policy cost

A separate sequential-mode HTTP run uses the identical three prompts, one
warmup and six measured requests per mode/case. Every request emits 64 tokens.
Precision can change greedy text, so these are policy tradeoffs rather than
matched-precision speedups; the alternating A/B qualification above is the
matched comparison.

| Workload | Default mixed, ordinary ms | FP32, ordinary ms | FP32, speculative ms |
|---|---:|---:|---:|
| Repetition | 1030.34 | 1212.87 | 217.96 |
| Explanation | 1029.41 | 1211.26 | 1153.54 |
| Code/list | 1034.20 | 1215.25 | 726.89 |

The explicit precision policy costs about18% without speculation. Its
speculative mode is still faster than default mixed on these repetition/code
cases, but explanation is about12% slower. This is why neither higher precision
nor speculation becomes the default.

The cumulative-state diagnostic reproduces the failure: after only two prior
parallel rows, both sequential and parallel evaluations from that accumulated
KV tie tokens1473/315 at21.328125, and argmax selects315. The original sequential
state had1473 at21.34375 and315 at21.328125. It demonstrates how earlier projection
rounding changes a later greedy choice even when a fresh same-state comparison
has identical argmax. This is not a cache-position or cross-slot corruption.

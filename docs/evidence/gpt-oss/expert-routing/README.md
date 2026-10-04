# Stable GPT-OSS expert grouping

Pinned GPT-OSS-20B checkpoint, A2, unchanged packed expert weights and matrix
arithmetic. Stable sorting groups flattened route IDs once per layer. One host
count vector defines slices; GPU token/route indices preserve the original
row-major order, including multiple routes from one token. No per-expert
`nonzero` operation is needed. Decode is unchanged.

Twelve GPU permutation cases cover random routes, all routes to one expert,
missing experts, boundary expert IDs and 1/16/129/4096 tokens. Full-model prefill
logits are bit-exact between old and new grouping at 16/129/513/789/4000 tokens,
using different KV slots, with identical eight-token greedy continuations.
All 250 CPU tests pass.

The warmed profiler records 769 -> 73 `cudaStreamSynchronize` calls and
768 -> 0 `aten::nonzero` calls. Waiting time includes GPU computation and must
not be interpreted as wholly removable CPU overhead. Profiling is attribution,
not latency evidence; overlapping operator/kernel totals must not be summed.

The latency evidence is four alternating warmed complete HTTP rounds with
16 output tokens, context 4096 and two slots. Both modes use the same fused
attention and size-dependent projection policy: compensated attention products
below 256 prefill rows, IEEE FP32 at/above 256. All response choices and usage
are identical. No simultaneous GPU workload ran during the benchmark.

| Prompt tokens | Concurrent requests | Original grouping | Stable grouping | Reduction |
|---:|---:|---:|---:|---:|
| 16 | 1 | 1.567 s | 1.413 s | 9.8% |
| 129 | 1 | 1.925 s | 1.782 s | 7.4% |
| 1025 | 1 | 3.650 s | 3.442 s | 5.7% |
| 4000 | 1 | 9.801 s | 9.605 s | 2.0% |
| 129 | 2 | 3.661 s | 3.272 s | 10.6% |

This improves host scheduling overhead; it does not accelerate the individual
expert matrix products or expand architecture/numerical qualification. The
long-context hidden-state investigation remains open in issue #41.

Reproduce with `scripts/verify_expert_routes.py`,
`scripts/benchmark_gpt_oss_routes.py` and `scripts/profile_gpt_oss_prefill.py`, using
the A2 UUID and the pinned checkpoint from issue #39. The profiler's initial
report predates the explicit CPU/CUDA event-type field; both reports retain
operator names and counts.

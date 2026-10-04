# Native GPT-OSS fused prefill attention

Checkpoint: `openai/gpt-oss-20b` at
`6cee5e81ee83917806bbde320786a8fb61efebee`. Device: NVIDIA A2.
Native MXFP4 expert storage, FP32 residual/KV/output, grouped expert prefill,
two request slots and unchanged CUDA decode graphs.

## Change

At 256 or more prefill rows, attention uses the pinned PyTorch efficient CUDA
operator and its log-sum-exp output. A learned sink adds only denominator mass,
so the ordinary attention output is multiplied by `sigmoid(logsumexp - sink)`.
Full attention partitions query heads to bound GQA expansion. Sliding attention
partitions query rows and trims the keys to the union of the rows' legal windows.
Neither constructs a context-squared mask. Small prefills and decode retain the
existing native kernel. The host passes its known integer slot directly; the
fast path does not read a slot ID back from the GPU.

The attention projections at these longer prefills use IEEE FP32 products with
one transiently expanded matrix. This repairs the observed nearly tied expert
routing case without expanding all resident weights or changing expert
projections. TF32-enabled global matmul policy is rejected, including changes
after model loading. `DLLM_GPT_OSS_ATTENTION=native` selects the bounded native
attention kernel while retaining the same IEEE attention-projection policy.

## Verification

- 72 original short layer/logit/decode/slot checks pass.
- 72 additional checks at 257/505-token prefixes pass against independently
  generated CPU Transformers FP32 layers, including complete 265/513-token traces.
- All five sensitive-prefix output checks pass. The previously wrong 789-token
  prefix now selects reference token 279; its logit maximum absolute error is
  0.000262. Native bounded/unbounded attention remains bit-identical under the
  repaired projection policy.
- Independent FP64 attention checks cover full/windowed attention, multiple
  prompt sizes, sinks from -100 to 100 and window boundaries. Maximum selected-row
  absolute error is below 8.35e-7. At 4K, incremental attention peaks are
  92,667,904 bytes (full) and 77,827,072 bytes (window), versus 101,163,008 baseline.
- Real HTTP at 4000 prompt tokens / 4096 context passes streaming, cancellation
  recovery, concurrent equivalence, graph reclamation on swap, host restore and
  explicit unload/reload. Peak allocation is 15,191,485,440 bytes.
- Long Harmony auto-function calls pass framing, JSON handoff, tool-result
  history, incomplete-call suppression, stream equivalence, cancellation recovery
  and concurrent call isolation at the same configured context.

## Matched complete HTTP measurements

Four alternating warmed rounds per workload, same resident checkpoint/context,
same IEEE FP32 attention-projection policy in both modes, same output length
(16), same decode graphs, no simultaneous GPU workload. All response choices and
usage are exactly equal. The baseline here is bounded native attention with the
repaired projection policy, not a different runtime or the original approximate
attention-projection policy.

| Prompt tokens | Concurrent requests | Native attention median | Fused attention median | Reduction |
|---:|---:|---:|---:|---:|
| 257 | 1 | 2.293 s | 2.148 s | 6.3% |
| 1025 | 1 | 4.495 s | 3.540 s | 21.2% |
| 4000 | 1 | 18.841 s | 9.743 s | 48.3% |
| 513 | 2 | 5.646 s | 4.983 s | 11.8% |

The concurrency-two case measures the complete pair of requests. These gains
apply to the measured GPT-OSS prefills; they are not decode speedups. All 248
CPU tests pass, including rejection of a changed global TF32 policy.

## Limits

This does not close [issue #41](https://github.com/filthyrake/dllm/issues/41).
Longer full-layer traces still fail the unchanged absolute hidden-state gate,
although the tested sensitive outputs are repaired. Independent CPU and CUDA
reference traces themselves also exceed that gate in late layers. Historical
failures, controlled substitutions and the reference comparison remain under
[the longer-context investigation](../context-4k/README.md). No threshold has
been relaxed and no near-tie routing epsilon has been added.

The fused operator is part of pinned PyTorch 2.6; this path is CUDA-only. This
work does not qualify 120B, larger contexts, multi-GPU execution, or other model
families. Component timings and HTTP timings are separate measurements.

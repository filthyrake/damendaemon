# OLMoE Q4 router-boundary diagnosis (#577)

Two artifacts bound the near-tie flip mechanism behind the retained single
failure (`olmoe-q4-model512-a2.json`: decode:498, slot 1, relative RMS 4.232%,
max abs 0.912, greedy agreement intact):

1. [Boundary-frequency study](olmoe-q4-router-boundary-study.json) (CPU
   reference, Q4_0 storage policy, FP32 arithmetic): across all 27,760
   layer/position rows of the 512-token golden trace, the top-9 routing
   probability gap around the cutoff is below 1e-8 for 2 rows, below 1e-7 for
   2 rows, below 1e-6 for 11 rows (396.25 per million rows), with minimum gap
   5.588e-9. The single observed flip is a rare boundary crossing, not one of
   many near-misses: the median cutoff gap is 1.09e-3, four orders of
   magnitude above the sub-ppm arithmetic noise.

2. [Origin localization](olmoe-q4-origin-localization-a2.json) (A2 under the
   guarded reserve gate, native Q4 executor vs HF reference on the 496-token
   prompt, layer 0, row-aligned per stage):

   | stage | relative RMS | max abs |
   |---|---:|---:|
   | embedding | 0.0 (exact) | 0.0 |
   | qkv input (norm1 out) | 8.840e-08 | 1.192e-07 |
   | qkv output | 5.906e-07 | 2.533e-07 |
   | attention core (o_proj in) | 1.147e-06 | 1.555e-07 |
   | o_proj output | 1.016e-06 | 1.341e-07 |
   | mlp input (norm2 out) | 9.179e-07 | 3.994e-06 |
   | mlp combined output | 5.067e-03 | 1.057e-03 |
   | sdpa vs o_proj input (native internal) | 0.0 (exact) | 0.0 |

   The first stage above the 1e-7 gate is the QKV projection (5.9e-7): the
   divergence originates in fp32 GEMM reduction-order differences between the
   CUDA and CPU arithmetic stacks on bit-identical fp16-storage weights, and
   stays near 1e-6 through attention into the router input (9.179e-7,
   reproducing the aligned diagnostic exactly). The sub-ppm router-input
   difference then crosses the eighth/ninth top-9 boundary at row 239 (gap
   1.3e-8), and the differing expert set amplifies the deviation to 5.1e-3 at
   the layer-0 MLP output — the mechanism behind the retained 4.232% decode
   failure. The internal SDPA consistency check is exact, ruling out capture
   or reshape error in the attention stages.

   The prefill last-token logits comparison for this prompt is 1.869e-3
   relative RMS (max abs 0.0333). An intermediate capture run reported a
   0.656 "whole-model logits" RMS; that number was a row-alignment artifact
   (a single packed-prefill logits row broadcast against all 496 reference
   positions) and is superseded by this run — it should not be cited.

Reproduction: `scripts/experiments/olmoe_q4_stage_capture.py` under
`guard_shared_cuda` with `DAMENDAEMON_QUANTIZATION=q4_0` on the retained
trace `results/olmoe-q4-router-f16-golden512.pt`.

## Verdict

The divergence is a genuine fp32-amplified near-tie: no stage diverges beyond
fp32 reduction-order noise (nothing exceeds ~1e-6 relative before the route
boundary), the boundary study shows near-ties are rare (396 per million rows
below 1e-6), and greedy agreement held throughout. Q4 OLMoE stays disabled in
normal registration; the Q4 path is numerically sound, but greedy single-token
agreement at long context is not guaranteed across independent arithmetic
stacks, and the retained 4.232% failure is that mechanism demonstrated.

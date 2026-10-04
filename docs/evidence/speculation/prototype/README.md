# Dense speculative verification prototype

Issue #37, informed by NInfer's separation of draft proposals from full-target
verification. No upstream code copied or external inference runtime introduced.
This isolated experiment uses the existing dense forward pass with several
queries at distinct causal positions sharing one KV slot. Every row evaluates
the complete target vocabulary. No shortlist changes the target distribution.

Pinned Qwen2.5-0.5B and Qwen3-0.6B model paths/revisions, precision and complete
outputs are recorded in the raw reports. A2,1024 context,64 generated tokens
ignoring EOS for the timing workload, one request, widths2/4/8. Three measured
runs per combination plus ordinary decoding controls; model and graphs warm.
Timing includes prefill, proposal search, GPU execution and token readback,
without HTTP. Oracle proposals come from the already-generated reference and
are an unattainable upper bound, not real speed claims.

All27 workload/width/proposal combinations per model produce the exact ordinary
greedy token sequence. Forced wrong-first proposals always reject; their stale
KV tails do not change later output. Nine additional gates per model cover real
EOS, deliberately early stop IDs, and the1024-context tail. Only dense greedy
CUDA decoding is addressed: no stochastic acceptance, recurrent state, quantized
weights, paged KV, or production scheduling was qualified by this prototype.

Width8 **real n-gram proposals**, speed ratio versus ordinary decoding:

| Workload | Qwen2.5 | Qwen3 |
|---|---:|---:|
| Explanation | 1.01× | 0.96× |
| Repeated sentence | 4.58× | 2.33× |
| Code/list generation | 1.59× | 1.35× |

Wrong proposals everywhere regress Qwen2.5 to about0.86–0.88× and Qwen3 to0.79×.
This motivates an opt-in serving path, rejection backoff and full HTTP testing.
The width8 oracle upper bound is about5.3–5.5× /5.0×, but requires a perfect
proposal source and is intentionally not presented as an achievable feature.

Reproduce with `scripts/probe_speculative.py` on the authorized A2. Later
serving measurements must include scheduling and stream-buffer costs rather
than substituting these favorable isolated numbers.

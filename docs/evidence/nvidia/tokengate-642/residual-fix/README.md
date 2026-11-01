# #642 residual — gated executor divergence root-caused and fixed (2026-10-31)

The PR #653 finding #3: our #641 gated executor produced 0/24 degenerate
counting output on the 24-row NIAH subset while upstream's (patched) harness
scored 5/24 (12 empty) on the same rows + checkpoint (step-1200, K=2000),
with full-context parity holding on both sides (10/24). This cell root-causes
and fixes the divergence. All runs guarded on the desktop A6000 (90 °C limit,
0 MiB reserve, UUID-pinned; guard reports retained alongside).

## Root cause (two real defects, both ours)

1. **Phase-2 was missing every MLP sublayer above the gate** — the decisive
   one. `_gated_phase2` ran only the mixer half (`linear_attention` /
   `_gated_attention`) for each layer above the gate layer; `_run_layer`
   (the standard path) is mixer half **plus** `_mlp_half`. Upstream's
   phase-2 loop runs full layers over the pruned sequence, and the MLP is
   rowwise, so kept-row MLP values are identical to a full forward — but
   only if the MLP actually runs. A 22-layer stack with no MLPs above
   layer 1 is exactly a uniform ~68 % hidden-state divergence born at the
   first layer above the gate.
2. **Decode-fold/bias-scale convention** — `TokenGate.decode_fold`
   collapsed the per-head key bias to its head mean and dropped the
   head-mean gamma factor; upstream scales the per-head bias
   `b_j[l,h] * mean_h(gamma_lh) * r_j` (their `_phase2_q_bias` /
   `decode_phase2_layer_tensors` — the code, not the comment above it,
   which describes an unshipped affine variant). Replaced by
   `key_scale`/`fold_bias` (per-head preserved, stored per-head in
   `gate_fold`), and `_gated_attention`'s prefill bias now carries the same
   scale, so prefill and decode share one bias convention.

## Bisection record (how the root cause was localized)

- **kept-set bisect** (`probe_642_bisect.py`): same kept counts (2013),
  ~10/2013 near-threshold flips per row — selection broadly agrees on this
  checkpoint (the flips are backbone-numerics noise; upstream's own kept
  set is not run-stable at K=2000 either).
- **kept-set forcing** (`probe_642_keptforce.py`): running OUR executor with
  upstream's exact kept set still produced 0/12 counting garbage — the
  divergence is inside our phase-2 forward, not selection sensitivity.
- **layerwise** (`probe_642_layerwise.py`, forced kept set): the phase-2
  entry (boundary) agrees at **0.71 % rel-RMS** (bf16-noise level, uniform
  across rows — article 0.71 %, suffix 0.75 %), while **layer 2 — the first
  linear-attention layer above the gate — already diverges at 68 %** and
  every later layer merely propagates it (0.85–1.08, norms growing apart).
- **cross-call** (`probe_642_crosslayer.py`): running OUR layer-2 forward on
  UPSTREAM's boundary reproduces the identical 68 % divergence (0.6820 vs
  0.6820 on our own boundary) — the boundary inputs are equivalent and the
  layer-2 forward itself is the defect site; `chunk_scan` is inactive
  (torch fallback both ways), killing the chunk-kernel hypothesis. Reading
  `_gated_phase2` against `_run_layer` then found the missing `_mlp_half`.

Why the #641 property suite missed it: "chunked gated == whole gated" is
self-consistent (both paths skip the MLP equally), the untrained-gate
fallback tests exit before phase-2, and the decode-continuation reference
inherits the same phase-2 state — no test compared phase-2 against a
layer-complete reference. `test_gated_phase2_carries_every_mlp_half` now
pins gated prefill to a manual layer-complete reference AND asserts the
MLP-less variant differs by >10 %.

## Verification (A6000, fixed tree, same 24 rows + checkpoint as every arm)

`tokengate-642-eval-fixed2.json` (guard `guard-eval-fix2.json`, clean):

| arm | before fix | after fix | upstream harness |
|---|---|---|---|
| gated K=2000 | 0/24 (degenerate counting) | **5/24** | 5/24 (12 empty) |
| adapted, full context (control) | 10/24 | **10/24** | 10/24 |

Generated text modes now match upstream's (`\n\nBased on the provided
text…`, `Answer: …`, thinking-prefixed); TTFT 3.16 s mean at K=2000, peak
3.72 GiB, wall 524 s for the guarded cell. The first pre-fix rerun
(`tokengate-642-eval-fixed.json`, fold fix only, MLPs still missing) stayed
0/24 with the same counting signature — both fixes are recorded, retained
as a finding that the bias convention alone was not the root cause.

## Honest limits

- The residual K-sweep rerun (K=100/500/1000 after the fix; the pre-fix
  table showed 0/24 at every K) was not executed — launch approval was not
  granted this session. K=2000 — the budget of the PR #653 finding — is
  the verified arm.
- Convergence is at the measured level (accuracy + generation modes +
  first-token distribution), not byte-exactness: the two executors differ
  in backbone kernels (ours vs HF reference), dtypes and their own
  run-to-run kept-set jitter.
- The checkpoint itself remains weak at K=2000 (5/24 vs the full-context
  10/24 and the base model's 22/24) — unchanged from #642's training record;
  this cell only fixes the executor divergence.

Artifacts: probe scripts (bisect / keptforce / layerwise / crosscall),
all gate/guard JSONs and logs in this directory.

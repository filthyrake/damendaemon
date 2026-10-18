# GPT-OSS 1025-prefix replay on the A30 — absolute-gate localization (#41, 2026-10-18)

## Purpose

The #41 long-context gates were previously run only on the A2. This run extends
the same unchanged gates to the **A30** (authorized labnode3 device
`GPU-70f6a991...`, both GPUs idle, `guard_shared_cuda`, no incumbent service)
to determine whether the retained late-layer failures are device-specific or
intrinsic. Reference/replay scripts now admit the authorized A2/A30 pair.

## Run

- Checkpoint `openai/gpt-oss-20b` pinned revision `6cee5e81...` (staged at
  `/home/damen/dllm-dev/models/gpt-oss-20b`; the prepared catalog file was
  copied to its post-rename name for the current scripts).
- Reference: independent Transformers 4.56.1 CUDA FP32 trace, TF32 off
  (`float32_matmul_precision: highest`), prefix 1025, 24 layers, generated on
  the A30 under `guard_shared_cuda` (guard `passed`, 163 s, peak temp 44 °C).
- Replay: damendaemon executor, current main (post fused-prefill repair:
  `ieee_fp32_prefill_256_plus` attention projections, bounded fused sink
  attention, triton-mxfp4 experts, FP32 residual), `--no-graph`, A30.
- Guard refused nothing; both GPUs returned to 0 MiB / 42 °C after the run.

## Result

37 checks total (24 layers, 1 full-prefill, 1 prefix, 8 decode, 2 slot-reuse,
1 slot-isolation):

| Check class | Result |
|---|---|
| Full-prefill / prefix / slot-reuse logits | **PASS** (rel RMS 3e-6, max abs 5e-5) |
| All 8 teacher-forced decode steps | **PASS** (greedy identical, max abs 8e-5) |
| Slot isolation | PASS |
| Layer checks 0–17 | PASS (max abs ≤ 0.354) |
| **Layer checks 18–23** | **FAIL** the unchanged absolute gate |

Failed layer checks: relative RMS **1.7e-5 – 2.8e-5** (gate: < 1e-2) but
max absolute **0.69 – 4.39** (gate: < 0.5).

## Analysis — the absolute layer gate is miscalibrated for this model

- The failure pattern reproduces the A2 result (late layers, tiny RMS, endpoint
  outputs clean) on a different GPU and an independent CUDA reference — it is
  not A2-specific hardware/reference variation.
- The hidden-state magnitudes are enormous: reference layer-18 mean |value| ≈
  65, layer-23 ≈ 196, max ≈ 4600 / 13300. At the measured 2e-5 relative RMS,
  the *expected* absolute error against |values| of ~65 is ~1.3e-3 per element;
  the 0.5 absolute gate is ~380x tighter than the model's own scale supports.
- 99.4% of layer-18 hidden-state components exceed the 0.5 gate magnitude —
  the gate is not a meaningful discriminator at these magnitudes; it is
  dominated by the value scale, not by divergence.
- Every functional check (final logits, prefix logits, teacher-forced greedy
  including the historically sensitive 789-token region) passes at 3e-6 RMS —
  four orders of magnitude inside the RMS gate.

## Conclusion for #41

The remaining "failure" is an **absolute-error layer gate that doesn't scale
with hidden-state magnitude**, not model divergence: relative error is ~2e-5
on layers whose typical magnitude is 65–200 (absolute tail up to 13300), and
all output-level checks pass with huge margin. The honest options are (a)
re-baseline the absolute layer gate relative to per-layer hidden magnitude
(e.g. max abs < k · mean|hidden|), or (b) close #41 as "output-level parity
qualified; internal-state absolute gate acknowledged as scale-miscalibrated"
with this evidence. Neither changes any output gate or hides a failure; the
retained historical traces stay in place.

## Files

- `replay-1025.json` — full 37-check report (A30, guard context in report)
- `ref-1025-guard-2.json` — reference-generation guard report

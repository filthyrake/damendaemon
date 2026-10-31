# Qwen3.8-27B vision — oracle + depth-27 gate calibration (#50 step 1-2, 2026-10-31)

The 27B snapshot @ `1d4bf0f2` is the full multimodal checkpoint
(`language_model_only: false`; 333 `model.visual.*` tensors, depth-27 /
1152-wide ViT — the same encoder family as the qualified 2B path, output
projection to the 27B text width 5120). The desktop A6000 is the first
fleet device able to host 27B text + vision together, so the #50
remaining product work (the "practical Qwen3.8+ image/text checkpoint")
starts here.

## Oracle (pinned transformers 5.18.0, CPU FP32 + CUDA strict-FP32)

`scripts/reference_qwen35_vision.py` ran unmodified against the 27B
checkpoint on the desktop:

- **CPU FP32** (`qwen38-27b-vision-cpu-fp32.json`, target .pt 380 MB,
  retained on the desktop at `/tmp/g50/` — .pt targets live outside the
  repo like the 2B-era ones): five standard fixtures (square,
  landscape, resized, larger, two-images); encoder outputs project to
  `[N, 5120]`, merged with text-position captures.
- **CUDA strict-FP32** (A6000, guarded, TF32 off —
  `guard-oracle-cuda.json`, 6.5 s): identical fixtures.
- Environment note: no fleet host had the pinned 5.18.0 anymore, so a
  dedicated target-dir install (`/home/damen/v-oracle-libs`) provides
  transformers 5.18.0 over the 3090 venv's torch 2.13.0+cu130 via
  PYTHONPATH — the 539 MB standalone torch wheel download from this
  host's mirror was impractically slow (retained as a recorded pitfall).

## Device-spread calibration — the depth-24 gates bound depth 27 with ~50× margin

`compare_vision_references.py` (the #573/#575 method — a gate may not be
tighter than the trusted reference's own device-to-device spread):

| metric | depth-24 (2B, #573) | **depth-27 (27B, this run)** |
|---|---|---|
| worst oracle device-spread RMS | 1.16e-04 (square block 23) | **1.26e-05** (square block 26) |
| worst abs / max\|target\| | ~5.35e-04 | **4.2e-05** |

The depth-27 encoder accumulates an order of magnitude LESS drift than
depth-24 did, and every fixture/stage passes even the older depth-12
absolute gate (max(0.02, 5e-4 × max)). **Conclusion: the existing
depth-24-calibrated gates (RMS 7e-04, absolute max(0.02, 2e-03 ×
max|target|)) apply to depth 27 with ~50× margin — no relaxation and no
re-calibration needed.** The encoder gates for the 27B are the same
numbers already qualified for the 2B path.

## Next (steps 3-5, recorded on #50)

A6000 vision-executor bring-up: an isolated probe admitting the
5120-wide text geometry + A6000 UUID exactly like the #68 Q4 probe
pattern (production `vision_policy.py` untouched), encoder checks
against this oracle under the existing gates, then the full-model image
gates (54-check suite + changed-image control) and the HTTP image
lifecycle at ctx ≤1024, ≤4 slots.

# #50: HOMER (Radeon 8060S, gfx1151) image serving qualification (2026-10-25)

The remaining "extend supported hardware" item from #50's current scope ran on
HOMER against the **same pinned references** the A2/M4/A30 gates consumed — the
transformers-5.18.0 CPU FP32 encoder reference (`results/qwen35-vision-cpu-fp32.pt`,
generated on this Mac) and the Q8_0 full-model/service reference
(`results/qwen35-images-q8-golden.pt`). Qwen3.5-0.8B only; the 2B depth-24 gates
were not run on this device.

## Gate 1 — standalone encoder (depth-12 calibrated bounds, unchanged)

`verify_qwen35_vision.py --device cuda` (HIP namespace) on the 8060S,
compensated CUDA-vision backend active (`compensated-cuda-vision`), TF32 off:

**75/75 checks passed** across all five fixtures (square, landscape, resized,
larger, two-images); worst relative RMS **1.849e-05** against the 1e-4 gate;
absolute gates per the #571 magnitude-scaled rule. Guard max **53.6 °C**.

## Gate 2 — full-model image/text logits (Q8_0 text)

`verify_qwen35_images_model.py --device rocm`: **54/54 checks passed**, worst
relative RMS **4.128e-03** against the 0.01 gate; changed-image control healthy
(native logit difference 0.892 vs reference 0.883 — image content actually
changes the output). Guard max **46.3 °C**.

## Gate 3 — real HTTP service lifecycle

`verify_vision_service.py --device rocm`: **20/20 checks passed** — independent
cases, warm requests, concurrency, streaming content+usage, disconnect
recovery with a changed image, malformed-input recovery, second model alias,
host restore, disk-reload-after-release, repeat-release with no allocation
growth. Guard max **45.8 °C**.

## Policy change (admission on merit, by name)

`vision_policy.validate_execution` now admits **AMD Radeon 8060S Graphics**
under `DAMENDAEMON_DEVICE=rocm`, with the same thresholds, geometry limits
(context ≤1024, ≤4 slots), and opt-in as every other device; unsupported ROCm
devices keep the explicit error (test covers admit + reject by name).

## Device-specific findings (disclosed, not silent)

1. **hipBLAS retains a 32 MiB persistent workspace after full unload** on this
   torch 2.11.0+rocm build. The A30 equivalent was identified via the #575
   CUDA allocator trace; no ROCm trace tooling exists yet, so the service gate
   discloses and tolerates 32 MiB explicitly and asserts no growth across the
   release cycle. A ROCm allocator trace is the follow-up that would tighten
   this to a identified-workspace bound.
2. `verify_qwen35_images_model.py`/`verify_vision_service.py` accepted only
   cuda/mps device strings; `rocm` added, mapping to torch's HIP device
   namespace while `DAMENDAEMON_DEVICE=rocm` carries admission/policy.
3. The 0.8B-run env requires the standard HOMER ROCm selection
   (ROCR_VISIBLE_DEVICES=0, HIP_VISIBLE_DEVICES=0, DAMENDAEMON_ROCM_MEMORY_GIB);
   otherwise memory admission rejects with the documented error.

Raw reports + guard reports on the host at /tmp/v50/ and mirrored in this
directory. Every cell ran under `guard_homer` (limit 85 °C, fans fixed@5);
worst observed temperature across all three gates: 53.6 °C.

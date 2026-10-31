# #50: NVIDIA RTX A6000 image serving — Qwen3.8-27B full multimodal (2026-10-31)

First fleet device holding the full 27B multimodal checkpoint
(`Qwen3.8-27B` @ `1d4bf0f2`: 64-layer 5120-wide qwen3_5_text backbone, 27B Q8_0
text ~27 GiB + depth-27/1152-wide ViT projecting to 5120). The desktop card
(NVIDIA RTX A6000, 44.5 GiB, sm86, torch 2.13.0+cu130) ran every gate under
the guarded shared-GPU gate (90 °C limit, 0 MiB reserve, UUID-pinned), TF32 off.

## Gate 1 — standalone encoder (depth-24-carried bounds, unchanged)

`verify_qwen35_vision.py --device cuda` vs the pinned transformers-5.18.0 CPU
FP32 oracle (sha256 `20b2cc4b…`, generated this cell series — see
`../qwen38-27b/README.md`): **150/150 checks passed** across all five fixtures
(square, landscape, resized, larger, two-images); worst relative RMS
**2.04e-05** vs the 7e-04 gate; `compensated-cuda-vision` backend.
Guard clean (`guard-gate1-enc.json`; the first launch attempt failed on a
staging-path miss and is retained as `guard-gate1-enc.attempt1-refused-path.json`).

## Gate 2 — full-model image/text logits (Q8_0 text)

`verify_qwen35_images_model.py --device cuda` vs the independent
Q8_0-text reference (`reference_qwen35_images.py --device cpu
--text-quantization q8_0`, transformers 5.18.0, golden .pt sha256
`473d308b…`, six cases incl. inverted/reversed changed-image controls):
**54/54 checks passed**, worst relative RMS **6.93e-03** vs the 0.01 gate,
greedy agreement on every check (`gate2-model-a6000.json`).

The `--messages` variant (inline PNG chat messages, model-owned
`VisionModelWeights`, `vision_policy` admission active):
**54/54 passed, identical worst RMS** (`gate2b-messages-a6000.json`; first
attempt failed on a missing PIL dependency in the run venv — fixed by
installing pillow into that venv, rerun clean).

## Gate 3 — real HTTP service lifecycle

`verify_vision_service.py --device cuda`: **20/20 lifecycle checks passed**
(`gate3-svc-a6000-final.json`, plain `python -m damendaemon serve` command,
two model aliases, host budget 63 GiB): independent cases, warm requests,
four-way concurrency, streaming content+usage, disconnect recovery with a
changed image, malformed-input recovery, second alias, host restore, disk
reload after release, repeat release with no allocation growth. Peak
`cuda_allocated_bytes` 32.1 GiB during the cycle.

**Persistent workspace finding (disclosed, same as A30 #579):** after full
unload the card retains **8,519,680 bytes — a single cuBLAS workspace block,
entirely `setWorkspaceForHandle`-framed** on this torch 2.13 build
(`qwen35-http-retained-allocations.json`; the diagnostic run
`gate3-svc-a6000-diag.json` identified it live). Root cause of the first
gate-3 failure: `diagnose_vision_server.py` only recognized the pre-2.13
`getNewWorkspace` frame name; fixed to match `verify_vision_service.py`'s
#575 dual-name identification. No unexplained growth across the release
cycle (repeat-release check passed).

## Policy change (admission on merit, by name and geometry)

`vision_policy` now admits the **64-layer 5120-wide** qwen3_5_text geometry
(exact pairing with the existing 24-layer 1024/2048-wide entries) and the
**NVIDIA RTX A6000** by name — same thresholds, context ≤1024, ≤4 slots, and
opt-in as every other device. `vision_qwen35.read_vision_config` enumerates
the depth-27/1152→5120 encoder geometry; `vision_gates(27)` carries the
depth-24-calibrated bounds (the measured depth-27 oracle device spread sits
~50× below them, PR #661). Other devices/geometries keep the explicit error.

## Honest scope

Still-image serving at context ≤1024, ≤4 slots, Q8_0 text (the FP16-text
27B variant is not cell-qualified). Long-context vision remains #60;
concurrency beyond the four-slot vision limit is separate from the #68 Q4
32K finding. The 380 MB oracle .pt targets and the 54 MB golden reference
stay outside the repo (desktop `/tmp/g50/`, sha256s recorded above);
HOMER untouched throughout.

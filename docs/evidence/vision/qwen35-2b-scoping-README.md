# Qwen3.5-2B vision — scoped, not admitted (#573)

Date: 2026-10-21 (UTC). Checkpoint `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` (dense text hidden 2048 / 24 layers;
vision tower depth 24 / hidden 1024 / 16 heads / out 2048; `deepstack_visual_indexes: []`).

## Verdict

**The 2B geometry is NOT admitted for experimental vision serving** —
`vision_policy.py` is unchanged and still admits only the 0.8B geometry. The
2B intermediate encoder gates fail in a way the depth-12 0.8B encoder never
did, and the failure is **below the reference stack's own device spread** at
depth 24. This issue closes on the #573 stopping condition: honest scoping,
no gate relaxation, no policy widening.

## The finding: depth-24 relative deviation exceeds its own gate's noise floor

Intermediate relative-RMS deviations (vs the trusted CPU FP32 oracle,
Transformers 5.18.0) accumulate with depth and are content-dependent — only
the `square` fixture is affected:

| Stack | square worst RMS | failing stages |
|---|---|---|
| A2 runtime, FP16 projection storage (`qwen35-2b-enc-a2.json`) | 1.70e-04 | blocks 19–22 |
| A2 runtime, FP32 projection storage (`qwen35-2b-src-a2.json`) | 2.92e-04 | blocks 16–23 + merged |
| M4 MPS runtime (`qwen35-2b-enc-mps.json`) | 1.12e-04 | block 22 |
| **Oracle itself, CPU vs CUDA** (`qwen35-2b-vision-oracle-device-control.json`) | **1.16e-04** | block 23 |

The oracle row is decisive: two runs of the SAME trusted reference
implementation on different devices diverge by 1.16e-04 RMS on square block 23
(max |activation| 5,866; absolute delta 3.14 → element-relative 5.35e-04, even
above the #571 absolute bound). Both #571 bounds were calibrated on the
depth-12 population; **no implementation can be held to an intermediate bound
tighter than the oracle's own device-to-device spread**, and the runtime's
worst (2.9e-04) is the same order and signature as that spread. The A2
FP32-storage run being WORSE than FP16 storage rules out storage rounding as
the cause; this is genuine depth-dependent accumulation in every
finite-precision stack.

## What passes (product-level evidence)

- **Full-model gate: 54/54 PASSED** (`qwen35-2b-model-a2.json`, guarded A2,
  triton-hybrid, batch capacity 3): all greedy choices match the independent
  reference across prefill + 8 teacher-forced decode steps on 6 prompt cases
  (square, resized, two-images, square-inverted, two-images-reversed,
  text-only-reused-slot); worst relative RMS 0.00488 (gate 0.01), worst
  absolute 0.1126 (gate 0.5).
- **Semantic controls healthy**: changed-image control — identical tokens
  with inverted pixels produce a native logit difference of 0.9425 vs the
  reference's 0.9482 (the runtime reacts to image content like the
  reference); reversed image order and slot reuse pass.
- Merged final features pass the #571 absolute bound in every runtime run.

## Path to future admission (out of scope here)

Recalibrate BOTH intermediate bounds for the depth-24 geometry from an
ORACLE device-spread population (the #571 method, but with oracle-CPU-vs-oracle-CUDA
as the passing population), then rerun the encoder gates. The golden
references stay on labnode3: `results/qwen35-2b-vision-cpu-fp32.pt`,
`results/qwen35-2b-vision-cuda-fp32.pt`,
`results/qwen35-2b-images-golden.pt`. The MoE vision checkpoints
(`qwen3_5_moe`, NAS 35B-A3B/397B) additionally need MoE text dispatch and
remain a separate ticket.

## Runtime change included

`damendaemon/vision_qwen35.py::read_vision_config` now enumerates BOTH vision
geometries (0.8B depth-12 and 2B depth-24) so the diagnostic and
reference-generation scripts can load the 2B encoder. **Serving admission is
NOT widened**: `vision_policy.py` still rejects the 2B text geometry, so no
serving path can reach the 2B vision stack with these weights.

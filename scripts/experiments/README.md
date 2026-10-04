# Inference kernel experiments

These are opt-in research programs, not serving backends. They do not participate
in runtime backend selection. Run only against the named, authorized test GPU.

## Compensated Tensor Core MXFP4 route (rejected)

`probe_mxfp4_tensorcore.py MODEL OUTPUT.json` compares the existing packed
projection against `mxfp4_tensorcore.py`, which pads one expert route to a 16-row
Tensor Core tile, expands packed values inside each K tile, and uses the existing
high/low FP16 input decomposition with FP32 accumulation. Checkpoint scales must
lie in the exact FP16 range. The probe validates the range before launching.

On the A2 this layout was numerically close but about 6.7–8.2× slower than the
existing chunked FP32 reduction for tested gate/up/down projections. It was
rejected, not wired into the executor. That result concerns this specific layout;
it does not establish that every Tensor Core approach is unsuitable.

Reproduce from the repository root with `PYTHONPATH=.` and the A2 UUID in
`CUDA_VISIBLE_DEVICES`, using the same pinned GPT-OSS checkpoint documented in
issue #39. The probe tests layers 0/23, one/three input tokens, top-4 routing and
both expert projections, including the input-split launch in graph timing.
Evidence: `docs/evidence/gpt-oss/normal-scales/mxfp4-tensorcore-experiment-a2.json`.

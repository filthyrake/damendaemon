# #63 gate 4: quantized execution on XPU — Q8_0 vs FP32 greedy agreement (passed, 2026-10-18)

## Implementation (PR #527)

The XPU bring-up policy was FP32-only (`xpu_policy.validate_model` rejected every
`quantization != 'none'`). This gate replaces that boundary with opt-in eager-torch
Q8_0 execution:

- Policy admits `q8_0` for dense Qwen3 and the qualified Qwen3.5-0.8B hybrid
  geometry (`hybrid_metal.eligible`: context <=1024, batch <=4). Q4_0, scaled RoPE,
  `weight_format` and `quantization_config` remain rejected. No default changed
  anywhere else; CUDA/MPS/CPU quantization contracts are untouched.
- New `damendaemon/quantized_torch.py`: eager Q8 linear — dequantize the packed
  int8/FP16-scale matrix to the activation dtype, then dense linear — mirroring the
  CUDA expand-path semantics with no resident dequantized copy and no Triton.
- `quantization.Linear` dispatches `xpu` tensors to the eager path before the CUDA
  Triton fallback (the uint8 Q4 branch stays unreachable on XPU by policy).
- Dense executor (`qwen2.Executor`): XPU eager Q8 is exempt from the
  native-CUDA-kernels-only rejection; `packed_workspace_bytes` budgets the largest
  transient FP32 dequant matrix instead of a resident copy.
- Hybrid executor (`hybrid.HybridExecutor`): XPU exempt from the
  native-kernels packed rejection; packed int8 weights keep packed storage on
  device (int8 + FP16 scales + dequant workspace budgeted) instead of the blanket
  FP32 promotion.

## Gate result (rocky10-desktop, B580, torch 2.10.0+xpu)

Qwen3.5-0.8B snapshot `2fc0636...` served twice at the hybrid-geometry limits
(context 1024, batch 4, no graphs, display reserve active on the live desktop):

- **Greedy agreement: 5/5 prompts byte-identical between FP32 and Q8_0**
  (`17 * 23 = 391.`, Paris, Yellow, counting, sky-is-blue reasoning).
- **Concurrent batch of 4 distinct prompts: correct outputs on both servers**,
  identical text FP32 vs Q8_0 (`concurrent_ok: true`).
- **Memory**: steady-state XPU allocation after the rounds — FP32 3.19 GB vs
  Q8_0 2.01 GB (packed storage saves ~1.18 GB); Q8_0 peak 4.08 GB during load,
  both inside 12 GiB with the 2 GiB display reserve.
- Sequential prompt wall times (capability only, never performance): Q8_0 eager
  dequant runs ~2x slower than FP32 (e.g. 2.0 s vs 1.1 s on the 24-token
  arithmetic prompt). Per-matmul eager dequant is the expected cost; no
  performance claim is made.

Two serve startups completed end to end (health, admission, generation,
telemetry, termination). Runner: `scripts/experiments/xpu_quant_gate.py`.

## Scope and honesty notes

- The hardware gate qualifies **hybrid Q8_0**; the host has no dense Qwen3-0.6B
  checkpoint (only Qwen3-8B, which exceeds the device with FP32-promoted
  embeddings under the display reserve), so dense Qwen3 Q8 admission is covered
  by CPU tests and the shared eager path, not by this hardware run.
- The first gate run (`xpu-quant-gate.json`) passed without the concurrent round
  (runner claimed it but did not implement it — docstring mismatch fixed in
  `3551aec`); the run was retained and the full gate rerun (`xpu-quant-gate-v2.json`).
- The earlier rejected `q8_0` cases in `test_xpu.py` became admissions; the
  Executor-level rejection case moved to `q4_0` (still rejected at policy).

## Files

- `xpu-quant-gate-v2.json` — final gate report (greedy 5/5, concurrent_ok, memory)
- `xpu-quant-gate.json` — retained first run (no concurrent round)
- `xpu-quant-gate-v2.serve-none.log` / `xpu-quant-gate-v2.serve-q8_0.log` — serve logs
- `xpu-quant-gate-v2.serve-*.log.manifest.json` — served manifests (via scp; not committed)

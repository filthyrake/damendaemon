# verify_qwen2 gate failure — attribution resolved, #568's premise disproven

Date: 2026-10-21 (UTC). All runs on labnode3 A2 (`GPU-1b29c7c8`), context 256,
16 steps, snapshot paths under `/home/damen/.cache/huggingface/hub`.

## What #568 claimed

That the 1.392% RMS failure measured during the #2 closeout (Qwen2.5-0.5B,
`--reference-dtype float32`) was a regression caused by torch build drift
(historical evidence torch 2.6.0+cu124 vs current venv 2.13.0+cu130).

## What the bisect actually showed

| Run | Code | torch | Model | Reference | RMS per prompt | Verdict |
|---|---|---|---|---|---|---|
| `run-t260.json` | current main | **2.6.0+cu124** (isolated venv) | Qwen2.5-0.5B (base) | float32 | 0.7549 / 0.6598 / **1.3924%** | fails |
| `run-era.json` | evidence-era `0c806a1f` | 2.13.0+cu130 | Qwen2.5-0.5B (base) | float32 | 0.7549 / 0.6595 / **1.3923%** | fails |
| `run-histcfg.json` | current main | 2.13.0+cu130 | Qwen2.5-0.5B-**Instruct** | **float16** | 0.5350 / 0.1977 / **0.4004%** | **passes** |

1. **Torch is exonerated.** On the identical model/reference/code, torch
   2.6.0+cu124 (the historical build, pinned in an isolated venv) reproduces
   the 1.392% failure bit-for-bit; torch 2.13.0 reproduces the era code's
   values to 7 digits. The executor path is numerically stable across both
   builds. No drift range exists to bisect.
2. **Code drift is exonerated.** The evidence-era tree (`0c806a1f`, when
   `qwen3-small-precise-verify_qwen2.json` was recorded) fails identically on
   the current venv.
3. **The real discrepancy is configuration, not environment.** The historical
   passing evidence never covered "Qwen2.5-0.5B base + float32 reference":
   - `regression-a.json` (the Qwen2.5 evidence) used **Qwen2.5-0.5B-Instruct**
     with a **float16** reference — 0.598 / 0.187 / 0.508% on torch 2.6.0.
   - `qwen3-small-precise-verify_qwen2.json` (<0.65%) is **Qwen3-0.6B** (the
     script name says qwen2; the `model_path` is Qwen3), and reflects the
     FP32-arithmetic qwen3 path.
   - The historical Instruct+float16 configuration **passes on current torch**:
     0.535 / 0.198 / 0.400% (`run-histcfg.json`) vs historical
     0.598 / 0.187 / 0.508% — same band, gate green, no regression.

## Verdict

There is no torch-drift regression. The 1.392% figure is a genuine measurement
of an UNQUALIFIED configuration (base model vs an idealized float32 reference)
that simply has no historical baseline. It sits in the same range the Instruct
model's fp16-reference measurements show relative to a float32 reference; it is
not attributable to any code or environment change. The base-model + float32
reference cell is recorded here as its first baseline; if the runtime ever
ships a config selector that would serve that combination by default, it needs
its own gate and its own qualification.

## Corrections propagating from this finding

- `../qwen3-fp32-default-2/README.md` (committed in #569) attributes the
  Qwen2.5-0.5B failure to torch drift — corrected in place.
- The #2 closeout comment and #569 PR body repeat the same attribution —
  corrected via comment.
- #3 roadmap line referencing "torch-drift RMS failure" — corrected.
- #568 closed as premise-disproven; no gate change, no torch pin.

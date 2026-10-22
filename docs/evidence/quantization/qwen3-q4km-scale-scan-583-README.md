# #583 Q4_K_M dense-GGUF scale scan — where does the <1% reference gate cross?

Date: 2026-10-22 (UTC). Instrument: `scripts/verify_gguf_reference.py` initial
suite, context 512, eight decode steps, both graph modes, TF32 off, guarded
shared-CUDA reserve gate on the labnode3 NVIDIA A30, torch 2.13.0+cu130.
Every threshold unchanged; failures retained; no retries with changed settings.
GGUF sha256 recorded per run report.

## The scale curve

| checkpoint | params | reference | worst relative RMS | greedy mismatches | gate |
|---|---|---|---:|---:|---|
| Qwen3-0.6B Q4_K_M (`qwen3-small-import`, rev c1899de…) | 0.6B | FP32 | 0.362% | 0 | PASS |
| Qwen3-1.7B Q4_K_M (`qwen3-medium-import`, rev 70d244c…) | 1.7B | FP32 | 0.239% | 0 | PASS |
| Qwen3-4B-Instruct-2507 Q4_K_M (NAS 2.50 GB) | 4B | FP32 | 0.365% | 0 | PASS |
| Qwen3-8B Q4_K_M (NAS 5.03 GB) | 8B | FP16 GPU | 1.903% | 0 | FAIL* |
| Qwen3-8B Q4_K_M — FP32 control | 8B | **FP32 CPU** | **0.573%** | 0 | **PASS** |
| Llama-3.1-8B-Instruct Q4_K_M (retained #581 checkpoint) | 8B | FP16 GPU | **14.069%** | 0 | FAIL |
| Llama-3.1-8B — FP32 control | 8B | **FP32 CPU** | **13.588%** | 0 | **FAIL** |

*Both 8B FP16 rows are bounded by paired FP32 controls: qwen3's apparent
failure was reference noise (the runtime passes); llama's failure is real
drift (13.588% survives a true FP32 reference; only ~0.5 points of the FP16
row was reference noise). The #43 gate cells are the FP32 rows.

Run reports: `qwen3-06b.json`, `qwen3-17b.json`, `qwen3-4b.json`, `qwen3-8b.json`,
`llama31-8b-f16.json` in this directory. Graph replay was bit-exact with eager
in every run. The A30/A2 two-device agreement for the llama checkpoint is in the
[#581 isolation note](../llama31-8b-q4km-isolation-README.md).

## Reading

- **Qwen3 dense Q4_K_M holds the <1% gate through 8B** — every qwen3 cell from
  0.6B to 8B passes against an FP32 reference. The apparent 1.9% failure at 8B
  was **FP16-reference arithmetic noise**: the paired FP32 control on the same
  fixture measures 0.573%. This is the documented golden-trace trap (fp16 HF
  references sit at the gate themselves) surfacing at 8B scale; future 8B-class
  ladder runs must record `--reference-dtype float32` (CPU-serial) or bound the
  fp16 reference explicitly.
- **Llama-3.1-8B fails on real drift, not reference noise**: 14.069% against the
  FP16 reference and **13.588% against the paired FP32 control** — the FP16
  reference contributed only ~0.5 points. The per-case profile is identical in
  shape (45→0.72%, 48→3.15%, 129→3.56%, 505→13.59% worst-step RMS), zero
  greedy mismatches on the official fixture. The checkpoint stays unqualified.
- **The failure is checkpoint-specific, not scale-general**: at the same 8B
  scale, on the same kernels and instrument, qwen3-8B passes at 0.573% while
  llama-3.1-8B fails at 13.588%. Combined with #581 (device-independent) and
  the prompt-length/content dependence, the llama failure is consistent with
  the accumulated-drift/near-tie mechanism family documented for OLMoE Q4
  (#577/#578) and qwen3 standard mode (#2) — a property of this checkpoint's
  quantization error interacting with long-context accumulation, not a kernel,
  device, or instrument defect. Per-stage origin localization (the #577
  playbook) is the identified path if it is ever pursued; no bound that admits
  13.6% is a meaningful drift gate.
- **Zero greedy mismatches in every 8B run** (both families): the drift has not
  flipped a greedy token on these fixtures; the llama #581 bespoke probe did
  observe one step-0 flip on its own fixture, so greedy stability is not
  guaranteed in general.
- **4B is a new gate-1 pass** for the #43 real-checkpoint matrix at a scale
  above the previous 1.7B ceiling. Full ladder qualification (runtime,
  tokenizer/chat, HTTP lifecycle) is the natural next #43 cell and is NOT
  claimed here.

## Reference-dtype bound at 8B

The 8B cells used an FP16-arithmetic HF reference because an FP32 8B reference
(32 GB) fits no GPU in this environment. Two bounds recorded:

1. **qwen3-8B direct control** (`qwen3-8b-fp32ctrl.json`): same four prompts,
   same fp16 weight rounding, reference executed in FP32 on CPU, runtime on the
   A30 teacher-pinned to the reference greedy path. Worst relative RMS
   **0.573%**, greedy agreement on all four cases, guard phase `completed`.
   This reverses the FP16 row's 1.903% for qwen3: the runtime passes.
2. **llama-3.1-8B direct control** (`llama31-8b-fp32ctrl.json`): identical
   method. Worst relative RMS **13.588%** (worst case the 505-token prompt,
   worst-step max absolute 1.782), greedy agreement on all four cases, guard
   phase `completed`. This confirms the llama failure as real drift: the FP16
   reference contributed only ~0.5 points of the FP16 row's 14.069%.

Incident note: the first llama control launch silently re-executed the stale
qwen3-hardcoded script copy on the host (its scp had been bundled inside a
consent-blocked command and never transferred). The duplicate was caught by its
bit-identical qwen3 result and the report's own `gguf` field, and the control
was rerun with an md5-verified script. The invalid duplicate output is retained
on the host as the incident record and is not evidence.

## Reproducing the controls

`scripts/experiments/q4km_fp32_reference_control.py` (added in this change) is
the repository copy of the control used for both 8B rows. For auditability: the
executed llama control script is byte-identical to the retained host copy
`/tmp/dd583_fp32_control.py` (sha256
`f2499fec2598892ddf591aacb3d1b0c54c444bb2c5cbe053c655e1e1f2826c37`); the
executed qwen3 control used an earlier hardcoded-variant of the same method,
since overwritten on the scratch host — its method is the one described above
and each run report records its own `gguf`, `reference` and `torch` fields for
verification. The repo copy is a method-identical refactor with argparse
plumbing and is the canonical form for future runs. Habit rule recorded from
the incident: verify the remote script hash before any guarded launch.

## Disposition

- No gate or threshold changed anywhere in this scan.
- **Qwen3 dense Q4_K_M is FP32-reference-qualified through 8B** on the gate-1
  numerical instrument (context-512 initial suite). Ladder completion (runtime,
  tokenizer/chat, HTTP lifecycle) at 4B and 8B is follow-up #43 work and is not
  claimed here.
- **Llama-3.1-8B stays unqualified, with attribution closed**: the failure is
  real drift (13.588% against a true FP32 reference), checkpoint-specific (not
  scale-general — qwen3-8B passes on the same instrument), and
  device-independent (#581). The path forward recorded for any future attempt
  is per-stage origin localization (the OLMoE #577 playbook), not gate
  recalibration: no bound that admits 13.6% is a meaningful drift gate.
- **Method requirement recorded for #43**: any 8B-class dense GGUF ladder cell
  must use `--reference-dtype float32` (serial/CPU) or carry an explicit
  fp16-reference bound — an FP16 GPU reference at 8B added >1.3 points of
  apparent RMS at qwen3 scale and ~0.5 points at llama scale, in opposite
  directions of consequence.
- Option (a) (8B-class tolerance recalibration) is **not needed for qwen3** —
  the existing <1% bound holds through 8B against a valid FP32 reference — and
  **rejected for llama** — a 13.6% drift population admits no meaningful
  bound. This answers #583 completely; no recalibration work is warranted.

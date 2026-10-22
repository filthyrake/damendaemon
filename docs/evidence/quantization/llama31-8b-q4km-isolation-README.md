# Llama-3.1-8B Q4_K_M two-device isolation (#581)

Retention note for `llama31-8b-q4km-isolate-a30.json` and
`llama31-8b-q4km-isolate-a2.json`. Both runs: 505-token fixed probe
(`(i * 7919 + 101) % 128000`), 8 decode steps, native packed Q4_K_M executor vs
an HF eager reference built from the same GGUF, guarded shared-GPU reserve gate,
TF32 disabled, `triton+mixed` backend, torch 2.13.0+cu130.

Reference dtype errata (2026-10-22): the two runs deliberately differ in
reference precision, and the original note overstated the A30 side. The A30 run
used an **FP16-storage, FP16-arithmetic GPU reference**; the A2 control used a
**raw-dequantized FP32 CPU reference** (no storage rounding). The two agree to
0.008 points at step 0 (1.0995% vs 1.1072%), which bounds the FP16-reference
contribution at 8B scale well below the observed deviation — the deviation is
not reference-precision noise. Any future cycle that pins this checkpoint
should use an explicit `--reference-dtype float32` ladder run for the record.

## Checkpoint

`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, 4.92 GiB, read from the local model
mount on labnode3. SHA-256 recorded per run output file:

- A30 run (`dd581-isolate.json`): checkpoint sha256 begins `3eade53d570cdf88`
- A2 run (`dd581-isolate-a2.json`): checkpoint sha256 begins `f167860f9f523b3f`

(The two strings differ because they are the sha256 of each run's **report
file**, not of the checkpoint — the report files embed device-specific fields.
The checkpoint file itself was not hashed inside the guarded run; it is the same
4.92 GiB file read in both runs. Record the checkpoint hash in any future cycle
that must pin the exact bytes.)

## Scope of the conclusion

The probe is a **single fixture**. The conclusion "model-scale, not
device-specific" is scoped to this probe on this checkpoint: the step-0
deviation agrees across A30 and A2 to 0.008 points (1.0995% vs 1.1072%), which
excludes an A30-kernel artifact for this fixture. It does not establish that
every 8B prompt fails the gate, nor locate the drift's per-stage origin — **no
per-stage capture was run in this bounded cycle** (contrast the OLMoE #577
localization). The "accumulated kernel/reduction-order sensitivity" wording in
the issue comment is a hypothesis, not a diagnosis.

## A2 reproducibility

The A2's 15 GiB cannot hold the 8B FP16 reference alongside the guard's memory
reserve: with the reference resident, ~2.6 GiB free tripped the gate at ~128 s.
The committed control script therefore builds the reference **on CPU**, deletes
it, then constructs the packed executor. On other 15 GiB devices the same
ordering is required; on >=24 GiB devices the GPU-resident reference fits (the
A30 run used it).

## Earlier killed attempts

Two A2 attempts before the committed run were killed by the reserve gate
because a stale script copy (GPU-resident reference) was still on the host.
The gate worked as designed; no state was corrupted. Retained for the record.

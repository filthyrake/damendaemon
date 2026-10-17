# Bounded Q5 precision diagnostic (#219 / parent #45)

The pinned historical checkpoint SHA and fixed48-token arithmetic prompt/eight
teacher tokens are in `oracle.json`. Independent HF eager CPU execution uses
upstream gguf.quants dequantization with FP16-rounded coefficients, exported
separately in FP16 and FP32 execution. One CPU/BLAS thread,16GiB process virtual
memory cap and600-second external timeout bound the export. CPU oracle logits
are hashed and remain in the isolated host experiment directory. These CPU
oracles do not replace the original CUDA-HF numerical contract.

The shared-A2 packed `triton+mixed` replay completed in34.638 seconds with
peak reserved1042284544 bytes. Maximum relative RMS versus CPU FP16 is0.6954%;
versus CPU FP32 it is0.2806%. Both comparisons pass the unchanged<1% RMS,
<0.5 absolute and exact-greedy gates across the fixed eight-step trace. This
new diagnostic does not reproduce the historical1.0454% Triton-versus-CUDA-HF
FP16 failure; different reference devices/accumulation paths must not be silently
conflated. #45 remains open.

`guard-transition-refusal.json` records a subsequent FP32 candidate attempt
refused before launch when the previous suite PID disappeared. The user
confirmed the long-running incumbent test suite changes model configurations
through load/unload transitions. Its next model must become healthy and its
new PID/headroom be admitted afresh; released transition memory is not treated
as permanently available. No incumbent restart or configuration change occurred.

The explicit FP32 candidate subsequently completed under the replacement suite
PID2973750 in36.257 seconds, peak reserved1115684864 bytes. Its maximum RMS is
0.5791% versus CPU FP16 and0.1746% versus CPU FP32; both comparisons pass. The
backend's generic label remains `triton+mixed`; the report's precision field
records the explicitly selected `DAMENDAEMON_LLAMA_PRECISION=float32` policy. No native
residual library is injected by this isolated diagnostic. Full candidate reports
and successful guards are retained. Final free GPU memory returned to4255MiB
and the incumbent remained healthy.

Conclusion: neither bounded candidate fails these CPU-oracle controls. FP32
reduces this diagnostic's error, consistent with the prior precision evidence,
but the historical CUDA-HF-FP16 failed cell is not a CPU-oracle result and remains
unresolved. No threshold, default precision, supported-model claim or residual
admission was changed. #219's bounded tooling/diagnosis is complete; #45 remains
open for its original same-device oracle boundary. A full expanded GPU oracle
would require a separately admitted larger allocation, not silently borrowing
more VRAM from the running suite.

## Same-device CUDA FP16 oracle boundary (#45, 2026-10-17)

The missing same-device oracle is now exported and retained: an independently
constructed HF eager Llama reference with FP16-rounded `gguf.quants`
dequantized coefficients, running in FP16 on the A2 itself via
`accelerate.cpu_offload` (isolated Accelerate 1.10.0), one BLAS thread, TF32
disabled, with forward hooks verifying every watched Linear/Embedding/
DecoderLayer executed on CUDA. Fixed 48-token arithmetic prompt, eight teacher
tokens, context 64 — identical arithmetic to the earlier CPU oracles.
`oracle-cuda.json` records `reference_device: cuda`, the `cuda_float16` payload
SHA `72defd05c6308253...`, 93.5 seconds and 1056964608 bytes peak reserved under
`guard-oracle-cuda.json`. The 4 MiB logits payload stays in the isolated host
directory, hashed, matching the CPU-oracle handling.

The packed `triton+mixed` replay then compared against this same-device oracle
under `guard-mixed-cuda-oracle.json`: maximum relative RMS **0.5171%**, maximum
absolute difference 0.05322265625 (step 5), and all eight greedy decisions
exact — inside the unchanged <1% RMS / <0.5 absolute / exact-greedy gates
(`mixed-cuda-oracle.json`). Peak reserved 1042284544 bytes.

This establishes #45's original same-device oracle boundary on the pinned
checkpoint: against a clean, independently constructed CUDA FP16 eager
reference on the same device, the shipped mixed K-quant kernels stay within the
gate on this workload. The historical failing cell
(`llama32-q5-k-m-reference-cuda-initial-a2.json`, 1.0343% against its in-run
reference) is retained and not erased; that comparison predates the residual
and range kernel fixes and used its own in-run reference execution. This
diagnostic does not attribute that cell step-by-step, claim other checkpoints,
contexts or backends, or make any performance claim. The 4 MiB oracle payload
is not committed, per the established evidence handling.

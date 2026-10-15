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

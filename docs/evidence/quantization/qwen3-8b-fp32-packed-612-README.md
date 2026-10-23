# Qwen3-8B Q4_K_M FP32-arithmetic packed execution qualified (#612)

`DAMENDAEMON_QWEN3_PRECISION=float32` (FP32 matrix/activation/KV arithmetic
over the packed Q4_K_M weights, FP16 matrix storage retained) is qualified
for the qwen3-8B Q4_K_M checkpoint at context 512 on the A30 — the path
that unblocks the 8B ladder cell's gate 4. All runs under
`guard_shared_cuda` (TF32 off) on the byte-identical checkpoint; the
FP32 HF eager reference ran on CPU (`--reference-device cpu`, the
cross-hardware oracle of PR #611).

## Qualification chain (A/B against the committed FP16 mixed results)

| instrument | FP16 mixed (PR #611 record) | FP32 policy (this cycle) |
|---|---|---|
| teacher, initial suite | 0.573% worst RMS | **0.105% / max_abs 0.021 — PASS** |
| teacher, benchmark workload (16/129/400 tok) | **5.13% / 0.946 — FAIL** | **0.593% / 0.102 — PASS** |
| HTTP lifecycle (3 prompts, stream + non-stream, usage) | blocked (teacher failed) | **byte-exact vs teacher — PASS** |
| layout gate (128 checks: batch 1/2/4, eager/graph, fixed/paged, chunked) | 3.05% chunked event (corrected contract) | **2.84e-05 worst RMS — PASS** |

Zero greedy mismatches in every measurement. The FP32 benchmark teacher's
per-case worst: 16 tokens 0.062%, 129 tokens 0.593%, 400 tokens 0.320% —
the prompt-length-driven FP16 drift (cuBLAS M-variance + FP16 attention
rounding at width 4096, per #585) is gone; the residual 0.59% is the
packed-decode quantization signature — prompt-length-flat (the 129-token
case is the worst while the 400-token case is better), i.e. per-position
decode quantization, not accumulation. It sits inside the
published-GGUF contract (1% / 0.5). The layout gate's chunked cell shows
the cross-shape GEMM difference collapses to 2.8e-05 at FP32 (cuBLAS
FP32 kernels are M-invariant at this magnitude), below even the original
strict contract — independent corroboration of #585's mechanism
attribution.

## How to serve the 8B in this configuration

`DAMENDAEMON_QWEN3_PRECISION=float32` — set before process start
(per-process environment opt-in), now in the shared-CUDA guard's
PASSTHROUGH allowlist so guarded qualification runs can set it. The
policy applies process-wide: a serving process that registers multiple
models applies it to every qwen3 executor, including checkpoints
qualified at FP16 mixed (the 4B) — mixed-precision fleets should not
share a process with this configuration. Contexts beyond 512 are
unqualified for this path. FP16 mixed remains the default for packed
qwen3 checkpoints (faster; qualified only through the 4B-class gate-4
workloads — the 8B's long-prompt drift record stands).

The ABBA comparator was not rerun at FP32 by design: its ladder role is
cross-engine timing and lifecycle exercise, neither of which is
precision-dependent, and both are covered by the #611 comparator record
(lifecycle green) and this cycle's HTTP gate (stream/non-stream/usage
byte-exactness at FP32).

## Scope, recorded exactly

Qualified for qwen3-8B Q4_K_M at context 512 on the A30
(torch 2.13.0+cu130, triton+mixed, native K-quant decode where rows<=4).
Not claimed: other checkpoints, contexts, devices, or the FP16 mixed
default path; no throughput claims (FP32 prefill is slower than FP16 —
the ABBA comparator record for this policy is open follow-up if needed).
The bounded-drift acceptance decision for FP16 mixed at width 4096
(#612's option 2) remains open as a policy choice; this cycle delivers
option 1.

Evidence: `qwen3-8b-fp32-initial-a30.json` (+guard),
`qwen3-8b-fp32-benchmark-a30.json` (+guard), `qwen3-8b-fp32-http-a30.json`
(+guard), `qwen3-8b-fp32-layout-a30.json` (+guard) — all in this
directory; FP16 comparisons in `qwen3-8b-teacher-benchmark-a30.json` and
the ladder README.

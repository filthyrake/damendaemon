# Qwen3-0.6B standard-precision gate: independent reproduction + layerwise localization (2026-10-18)

## Result

The #2 standard-precision two-position failure is reproduced deterministically
on a freshly generated, independent golden trace (Qwen3-0.6B snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`, context 2048, teacher-forced,
A2 `GPU-1b29c7c8`, triton kernels, no CUDA graphs):

| Mode (`DAMENDAEMON_QWEN3_PRECISION`) | Checks | Failures |
|---|---|---|
| `standard` (fp16 compute) | 84 | 2 — decode event 8 slot 1 **1.1941%**, decode event 11 slot 0 **1.4159%** |
| `float32` (promoted) | 84 | 0 |

The two failing values match the historical evidence
(`qwen3-small-precise-verify_stress.json`: 1.1941% and 1.4159%) exactly —
same magnitude, same event/slot pairing — confirming a deterministic
divergence, not run-to-run noise. Greedy choices match on every event
(`greedy_match: true`); the gate failures are logit-RMS only.

## Layerwise localization

`layers-run.log` (script `scripts/diagnose_qwen3_layers.py`) compares the
executor residual stream per layer against the HF FP32 reference on the
trace: drift accumulates gradually across the stack (absolute-error jumps of
+0.03–0.10 spread over layers 10–26; relative RMS peaks ~2.9% mid-stack
before the growing residual norm dilutes it). There is no single failing
layer or op — the error is distributed fp16 arithmetic accumulation in the
dense projections / fp16 KV / fused decode kernels, consistent with the
issue's "accumulated mixed-precision residual error" framing.

## Methodology caveat (new)

Golden traces generated with the default `--reference-dtype float16` store an
**HF fp16 reference**, which itself sits at the 1% gate on some events: the
fully promoted `float32` executor fails 2/84 checks (1.009%, 1.778%) against
that fp16 reference while passing 84/84 against a true `--reference-dtype
float32` reference. Reproductions of this gate must generate references with
`--reference-dtype float32`. The near-boundary 1.005% values seen against the
fp16 reference are reference noise, not the standard-mode divergence.

Also recorded: a llama-style "mixed" policy (fp32 residual + rotary over
fp16 dense/KV) is a no-op for Qwen3 — `Executor.precise` is already True for
qwen3 in every mode (`damendaemon/qwen2.py`), so the FP32 residual stream and
rotary are already in the standard path; the divergence comes from the
remaining fp16 compute (dense projections, fp16 KV cache, fused decode
attention/MLP kernels).

## Conclusion

The precise boundary from #2 stands, now independently reproduced: the fp16
standard path accumulates distributed residual error past the 1% gate on
specific long-decode events while remaining greedy-correct; the promoted
float32 path passes the same trace. Correcting the source means removing
fp16 from the accumulate-heavy decode ops as a set — no single-kernel fix —
which is the already-qualified float32/compensated tradeoff.

## Files

- `f32ref-std.log` / `f32ref-std-guard.json` — standard replay vs FP32 reference (2 failures)
- `f32ref-f32.log` / `f32ref-f32-guard.json` — float32 replay vs FP32 reference (clean)
- `layers-run.log` / `layers-guard.json` — per-layer residual drift vs HF FP32
- `mixed-run.log` — mixed-mode bit-identical no-op demonstration
- Guarded runs only (`guard_shared_cuda`, A2, 900–1200 s caps)

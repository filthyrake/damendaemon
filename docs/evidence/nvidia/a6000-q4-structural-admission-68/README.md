# A6000 Q4 structural admission — production serve verified (#68 re-scope, 2026-11-01)

**Result: PASS.** Plain production `damendaemon serve` — no probe wrapper,
no patched validate — admitted the Q4-compensated Qwen3.8-27B at 32K
context on the A6000 and completed the pinned workload shape. The
device-name lock ('NVIDIA A30') and the gfx1151 arch string are retired
from production admission; guarded run exit 0.

## What changed (branch q4-structural-admission)

`hybrid_q4_policy`:
- CUDA admission is structural: the fused Triton hybrid kernels import AND
  a real CUDA device is present (`_cuda_q4_admits`). The retired name pins
  were proxies for "hardware we had measured" — the A6000 carries the same
  qualification (C1/C2 capacity, lifecycle, cancellation cells).
- The experimental context extension (512..32768 via
  `DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT`) follows admission instead
  of the gfx1151 string — which had already exceeded its own host's 4K
  qualification, so the string was never a capability boundary.
- The #536 two-device pipeline pair anchor and the distributed-chain
  anchor are intact (they encode topology, not device names).

Per the "things should not be locked to names" directive — admission
locks on capability, never identity strings. This dissolves the
"Q4-ROCm decision": any CUDA host with the kernels runs Q4; hosts without
them get a clear structural refusal.

## Measured (guarded, reserve 0 / thermal 90 / timeout 1800)

- Production serve admitted and became healthy with
  `DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4=1`, compensated precision,
  `DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT=32768` — the exact
  configuration the probe used to patch around (October's attempt-1 400
  refusal was this admission, before the probe existed).
- Warmup 512/8: first output 170.5 s.
- Measured 32K/64: first output **154.2 s** (C1 gate <= 600 s), 380 chars,
  coherent output.
- `/v1/models/status` reports q4_0 residency through the production
  endpoint.
- The standing 32K prefill OOM-retry wave appears in the serve log as
  expected (capacity finding, unchanged — eager prefill attention
  workspace growth, tolerated via allocator retry).

## Tests

64 passed + 1 skipped + 28 subtests: the reworked admission suite
(structural admit/refuse, execution bounds retained — opt-in, context
bound, batch <= 4, no graphs, qwen3_5_text only, compensated only,
pipeline/chain anchors), the context-extension suite (rocm and cuda
admit, xpu/mps refuse, bounds enforced), plus the layer-stream and
host-memory neighborhoods. One flake fixed in the #677 scale-rebind test:
a released tensor's id can be recycled by a new clone, so
"stale-id-absence" is unprovable by identity — the test now asserts the
real invariant (binding keys exactly match currently resident scaled
tensors).

## Artifacts

| File | sha256 |
| --- | --- |
| verify.json | b810d4ae8ba316e60382c8cd3392e274d0d00a6d67504673c453fc8ee12a3418 |
| guard-v.json | d08b0c32b2f47ee8600f1fa918f565570e530911ddc430b9304242a9ccc85d9f |
| verify.serve.log | 529bce2ef460720b5de34cb68abee30f4a2712c5d48bf2f624f20a42283f1cc7 |

The `hybrid_q4_a6000_probe` module remains for its pinned-workload
runners (UUID-pinned experiments); production no longer depends on it.

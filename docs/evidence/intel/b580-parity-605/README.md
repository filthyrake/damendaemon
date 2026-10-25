# #605 B580 fleet parity: matched baselines, graphs probe, SLO cell (2026-10-25)

rocky10-desktop only; the B580 kept driving the host display through every
cell (display reserve untouched; free telemetry recorded per cell).

## Gate 4 — matched baselines (Qwen3.5-0.8B, identical prompts/protocol, greedy)

| arm | device | served precision | torch | context | latency (64 tok) | throughput (256 tok) |
|---|---|---|---|---|---|---|
| A | Arc B580 | Q8_0 (opt-in qualified) | 2.10.0+xpu | 1024 | **65.3 / 63.5 ms/tok** | **16.1 tok/s** |
| B | RTX 3090 | FP32 | 2.13.0+cu130 | 2048 | 5.53 / 5.17 ms/tok | 308.3 tok/s |

Losses retained, not normalized: the 3090 is ~12× faster at latency and ~19×
at throughput on this fixture. The arms differ in served precision (Q8_0 is
the XPU-qualified format; FP32 is CUDA's native) and context (XPU hybrid Q8
admission caps at 1024) — both are the devices' qualified configurations,
recorded per row. Output heads match across arms for the identical greedy
prompts (Q8 byte-identical property from #63 applies to XPU self-consistency,
not cross-vendor FP32 equality; heads agree because the fixture is deterministic).

Operational findings: the XPU serve **segfaults without the Level Zero loader
prefix** (`LD_LIBRARY_PATH=/home/damen/b580-dev/ze-loader/lib64` required —
the documented recipe, now load-bearing for startup); quantization is env-driven
(`DAMENDAEMON_QUANTIZATION`), not a serve flag.

## Gate 1 — prefill graphs on XPU: REJECTED with the reason visible

torch 2.10.0+xpu exposes **no graph capture API**: `torch.cuda.CUDAGGraph` is
a dummy base class ("Tried to instantiate dummy base class CUDAGraph"),
`torch.xpu` has no `XPUGraph`/graph module, and `make_graphed_callables`
routes through the same dummy. Probed all three live on the B580.
`DAMENDAEMON_PREFILL_GRAPHS` stays rejected for Intel; the policy reason
already printed by `xpu_policy` remains accurate. Re-probe when a torch XPU
release ships a real capture API.

## Gate 2 — split/pipeline participation: prerequisites verified, two-vendor execution is the cross-vendor child's gate

XPU-side admission verified live: device admission, mem_get_info telemetry
beside the display reserve, host-mediated staging round-trips. **Material
finding: the XPU torch build is not compiled with CUDA** (`torch.backends.cuda
.is_built() == False`), so a two-vendor XPU+CUDA chain cannot run single-process
with this venv — the host-staged two-process chain is the required topology,
and that execution cell belongs to the cross-vendor child's gate. No policy
change is landed for split admission without its execution cell.

## Gate 5 — concurrency SLO cell at the admitted scope (batch ≤ 4)

Four concurrent 128-token streams, Qwen3.5-0.8B Q8_0 @ context 1024:
walls 11.355/11.356/11.357/11.357 s (spread **0.002 s**), all correct,
**45.1 aggregate tok/s**. Fairness at batch 4 is essentially perfect on the
B580's scheduler path; aggregate throughput ≈ 2.8× the single-stream 16.1 tok/s.

## Gate 3 — multi-XPU enumeration: NOT exercised (single-device host)

`xpu_policy` keeps `device_count() != 1` rejected; the desktop has one B580,
so no relaxation cell ran and the line stays unchanged (policy reason visible).

## Summary against the stopping condition

- Matched baselines: **done** (gaps retained).
- SLO-shaped concurrency cell: **done** (batch 4, fairness + aggregate).
- Prefill graphs: **explicitly still-rejected** (no capture API in the build).
- Multi-XPU: **unchanged/rejected** (no second device to qualify).
- Split participation: prerequisites only; the two-vendor execution cell is
  the cross-vendor child's gate — no policy relaxation landed here.

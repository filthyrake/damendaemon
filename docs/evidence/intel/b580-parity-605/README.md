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

## Gate 2 — split/pipeline participation: ADMITTED via the host-staged chain; single-process layer split stays CUDA-only by necessity

Disposition updated 2026-10-27 (follow-up pass): the prior gate-2 verdict
(PR #631, 2026-10-25) deferred the policy side, stating "no policy change is
landed for split admission without its execution cell" — written before
connecting to the cross-vendor child's closure: the two-vendor execution cell
**had already run and passed** (#600, closed 2026-10-23, merged as PR #614):
the B580 executed as a real chain stage
(Qwen3.5-0.8B `q8_0`, ctx 512, slots 2 — XPU rank 1: layers 12–24 + head,
eager FP32) over mutual-TLS `HostStageChannel` with greedy tokens identical
across all ten configurations and bit-exact boundary digests
(`docs/evidence/multi-gpu/cross-vendor-cell-600/`).

Policy + admission outcome (this issue's side):

- **The qualified topology needs no policy relaxation**: the #600 worker env
  (`DAMENDAEMON_EXPERIMENTAL_XPU=1`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`,
  `DAMENDAEMON_KERNELS=torch`, Level-Zero loader prefix) passes
  `xpu_policy` unchanged on current main — verified live by
  `gate2-admission-probe.json` (re-run on this branch, real B580,
  torch 2.10.0+xpu): `validate_environment` + `validate_runtime` admit the
  chain worker, and `available_memory` budgets 8.232 GiB of the live
  10.232 GiB free beside the 2 GiB display reserve.
- **`DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT` stays rejected — by necessity, not
  arbitrarily**: a single process cannot hold both vendors' torch builds
  (`torch.backends.cuda.is_built() == False` under the XPU venv, probed
  live), so the single-process layer split cannot include an XPU rank on any
  build. The rejection message now names the qualified alternative — an XPU
  rank joins fleets through the host-staged two-process chain — instead of a
  bare "does not support", which read as excluding split participation
  entirely (the opposite of what #600 qualified).
- `partition_policy`'s "layer partition requires CUDA" is correct and
  untouched: it guards the single-process CUDA split specifically.

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
- Split participation: **admitted via the host-staged chain** — the #600
  cell (merged, #614) qualified the B580 as a real chain stage and the
  existing policy admits that env unchanged (live probe); the single-process
  layer split stays rejected by build necessity, now with the qualified
  alternative named in the policy message.

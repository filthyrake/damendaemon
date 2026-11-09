# GLM-5.3 q4 fleet execution — rung 3b: the CUDA⇄ROCm chain cell (#719)

2026-11-08, rocky10-desktop A6000 (coordinator, torch 2.13.0+cu130, guarded
thermal 90 / reserve 0 / 1500 s) ⇄ homer 8060S (worker, torch
2.11.0+rocm7.13, HIP 7.13.26162, guard_homer 80 °C + in-cell CoolingGuard,
full homer procedure applied: strix-fan-daemon stopped, fans fixed/5 with
RPM readback, sheena stack stopped/verified-freed before and restored after,
the 9481 firewall rich rule added permanent and removed at cleanup).

## Result — ALL GATES PASS (attempt 2, `coord2.json` + `evidence2.json`)

**The first real cross-vendor pipeline execution in the fleet**: the
production `HostTensorChannel` (raw-byte framing over authenticated TLS,
rank/version/endian/identity handshake) carries real GLM-5.3 activations
between a CUDA executor host and a ROCm executor host — a genuine
torch-version AND vendor skew crossing on one wire.

Cell: coordinator builds the real layer-6 q4 FFN bank (256 experts + shared
+ real router/bias, 5.86 GB peak reserved) on the A6000, computes its output
on 8 seeded tokens, sends both across; the worker builds BOTH its layer-10
q4 bank and shipped-fp8 oracle bank (15.44 GB peak reserved) on the 8060S,
computes both over the boundary-chained input, returns q4 + oracle + an
echo. Stage identities derive from one deterministic shared `HostTopology`
(config.json + boundary-legal full-coverage ranges [[0,6),[6,78)]).

1. **Channel integrity**: the echoed tensor is **bit-identical** to the
   sent tensor — the CUDA→ROCm→CUDA round trip perturbs nothing across
   vendors and torch versions ✅
2. **Chained parity**: worker q4 vs its own fp8 oracle over the chained
   input — rel RMS **0.1176**, max_abs 0.120 × absmax (gate 0.15) ✅
3. **Finiteness**: every tensor on both ends finite ✅
4. **Worker mirror**: the worker's own gate computed the identical
   0.11764183640480042 rel RMS on gfx1151 that the coordinator computed
   from the received raw tensors on the A6000 — the two vendors agree on
   the same delta to the last fp64 digit ✅

No model, generation, throughput or SLO claim — this qualifies the
mixed-vendor pipeline boundary (tensors + framing + TLS + executors, end to
end).

## Attempt 1 retained (all four artifacts)

Both sides failed identically at first channel use: `host tensor channel
is not ready` — the bespoke runners skipped the channel's explicit
`handshake()` (the production pipeline always calls it; the readiness state
machine refused the transfer, which is exactly its job). Both banks had
already built cleanly on both GPUs. Fixed by calling `handshake()` before
the first transfer; both scripts' fixed copies are retained here.

## Files

- `coord2.json` / `guardc2.json` / `coord2.log` — the passing coordinator
- `evidence2.json` / `guard2.json` — the passing worker (mirror gates)
- `coord-attempt1.json` / `guardc-attempt1.json` / `worker-attempt1.json` /
  `guard-attempt1.json` — the retained handshake failure
- `glm_q4_chain_{common,coordinator,worker}.py` — the runners as launched

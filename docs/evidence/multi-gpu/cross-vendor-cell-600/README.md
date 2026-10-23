# Cross-vendor two-rank cell: RTX 3090 (CUDA) + Arc B580 (Intel XPU) — #600

Date: 2026-10-23 · Host: `rocky10-desktop` (10.40.100.30) · Script:
`scripts/experiments/cross_vendor_worker_cell.py` (commit on this branch)

## What was qualified

One model (Qwen3.5-0.8B, snapshot `2fc0636`, `q8_0`, context 512, slots 2)
executed as **two ranks on two vendors in two processes**:

| Rank | Process / venv | Device | Layers | Compute |
|------|----------------|--------|--------|---------|
| 0 (coordinator) | `3090-damendaemon/venv` (CUDA torch) | `cuda:0` (RTX 3090) | 0–12, embedding | fp16 + Triton |
| 1 (worker) | `b580-dev/venv` (XPU torch) | `xpu:0` (Arc B580) | 12–24, output head | eager FP32 |

Ranks speak the existing `HostStageChannel` protocol (`damendaemon-host/1`
ALPN) over mutual-TLS loopback sockets through `DeviceHybridStage` CPU
boundaries. Boundary activations are FP32 on the wire regardless of each
rank's compute precision (residual stream is FP32 by design, so fp16
compute widens exactly at the boundary). A CUDA torch and an XPU torch
cannot coexist in one process, hence two processes pinned to separate
venvs.

## Controls and verdicts (`compare.json`)

- **`cuda_protocol_control`: pass** — CUDA-only in-process pair matched the
  CUDA single-device executor token-for-token (protocol adds no divergence).
- **`xpu_protocol_control`: pass** — same for the XPU-only pair.
- **`wire_digests_match`: pass** — SHA-256 digests of every FP32 boundary
  activation the coordinator sent equal what the XPU worker received:
  the authenticated transport is bit-exact.
- **`cuda_boundary_bit_exact`: pass** — the mixed pair's CUDA-boundary
  digests equal the CUDA-only pair's: the XPU half changes nothing upstream.
- **`token_agreement`: 3/3 prompts, all 10 configurations** — the mixed
  3090+B580 pair produced greedy tokens identical to: single-CUDA
  (fp16+Triton), single-XPU (eager FP32), CUDA-only pair, XPU-only pair.
  FP16 on one rank and FP32 on the other still agree because both widen
  to the same FP32 boundary.

## Timing (informational only — not a performance claim)

Wall time for 3 prompts × 12 greedy tokens each: single-3090 ≈ 1.0–1.2 s;
single-B580 ≈ 0.9–1.6 s; any two-rank pair ≈ 3.3–4.2 s (loopback TCP
boundary serialization dominates; per-boundary blocking copies are the
known same-host behavior). Boundary transfer accounting per rank is in
`coordinator.json`/`worker.json` (`stats`: `bytes_in`, `bytes_out`).

## Reproduction

On the desktop host, with the worktree staged to `/tmp/damendaemon-600`:

```bash
# TLS material (once)
PYTHONPATH=. 3090-damendaemon/venv/bin/python -m scripts.experiments.cross_vendor_worker_cell \
  certs --dir /tmp/damendaemon-600-cell/tls

# XPU rank (B580 venv; local ze-loader prefix per b580-bringup-63 evidence)
LD_LIBRARY_PATH=/home/damen/b580-dev/ze-loader/lib64:$LD_LIBRARY_PATH \
DAMENDAEMON_DEVICE=xpu DAMENDAEMON_EXPERIMENTAL_XPU=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
DAMENDAEMON_KERNELS=torch b580-dev/venv/bin/python -m scripts.experiments.cross_vendor_worker_cell \
  worker --model <snapshot> --tls-dir /tmp/damendaemon-600-cell/tls --listen-port 8420 --output worker.json

# CUDA rank (3090 venv)
DAMENDAEMON_DEVICE=cuda CUDA_VISIBLE_DEVICES=GPU-24ef3d4c-… \
3090-damendaemon/venv/bin/python -m scripts.experiments.cross_vendor_worker_cell \
  coordinator --model <snapshot> --tls-dir /tmp/damendaemon-600-cell/tls \
  --connect-port 8420 --output coordinator.json

# Controls + verdict
… single / pair roles per vendor, then: compare --results single-cuda.json single-xpu.json \
  pair-cuda.json pair-xpu.json coordinator.json worker.json --output compare.json
```

Environment notes: the B580 process needs the local Level-Zero loader
prefix and segfaults under a plain `setvars.sh` environment (see
`docs/evidence/intel/b580-enumeration-fix-2026-10-06.json` and
`b580-bringup-63`); the 3090 was idle and pinned by UUID.

## Scope

Bounded qualification cell: two ranks, two vendors, one host, loopback
transport, greedy agreement. Not qualified here: serving integration,
multi-prompt concurrency SLOs, cross-host transport (10 GbE), RDMA, and
any performance/bandwidth claim. The script asserts its own evidence
(`__debug__` guard) and aborts on any control mismatch.

# #609 MPS × CUDA-graphs interaction: risk retired on the desktop 3090 (2026-10-25)

## Question

PR #626's decision cell used `--no-graph` and recorded the CUDA-graphs × MPS
interaction as a known untested risk. This cell closes it.

## Cell (desktop 3090, driver 615.71.09, Qwen2.5-0.5B, graphs ON)

| phase | capture | correctness | decode (128 tok) | per-process memory |
|---|---|---|---|---|
| graphs, MPS | **works** | correct (greedy identical) | 2.34 ms/tok | via mps-server (30 MiB) |
| graphs, no MPS | works | correct | 1.98 ms/tok | 1518 MiB |
| 2× graphs under MPS | works ×2 | both correct | 2.03 / 2.01 ms/tok | 1518 MiB each + 30 MiB server |

## Findings

1. **CUDA graph capture and replay work under MPS.** The #609 known risk is
   retired: co-tenant damendaemon serving can keep graphs enabled behind the
   MPS server. Both concurrent serves produced identical greedy output and
   served concurrently at ~2.0 ms/token each.
2. **Single-client decode cost of the MPS indirection: ~18%** (2.34 vs
   1.98 ms/token, one 0.5B tenant). The concurrent phase shows the per-tenant
   cost back near baseline (~2.0 ms/token) — the indirection is fixed-cost,
   not per-tenant-scaling.
3. **Graphs-enabled processes under MPS attribute ~1518 MiB per client**
   (vs 1386 MiB without graphs): capture pools add ~130 MiB per tenant.

## Operational findings for co-tenancy scripts

- `CUDA_VISIBLE_DEVICES=GPU-<uuid>` **fails under MPS when the control
  daemon's state is stale** (client sees 0 devices → `require_gpu` raises
  "exactly 1 CUDA GPUs must be visible"). A fresh `nvidia-cuda-mps-control`
  start with clean pipe dirs fixes it. Cells must kill any live MPS daemons
  and recreate pipe dirs before starting clients.
- Conversely, running a **non-MPS phase while an MPS daemon is alive with a
  deleted default pipe dir** also yields 0 devices. Teardown order matters:
  stop the daemon, then delete pipe dirs, in every phase transition.

Files on the host at /tmp/g609/: `graphs-{mps-graphs,plain-graphs,mps-two}.log`.

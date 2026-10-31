# #664 disconnect diagnosis — the "wedge" was observability, cancellation works (2026-10-31)

The #663 cancel cells concluded: (1) a mid-decode disconnect leaves a
zombie request holding its slot, and (2) at batch 1 a disconnect wedges
the whole server via a growing-allocation OOM storm. This diagnosis ran
three instrumented cells (D1/D3 at 32K, D1 at 4K, plus status polling
through a full 32K generation) and **corrects both conclusions**:

## Correction 1 — client disconnect is detected and the slot is released

D1-decode-32K (pre-fix, `d1-decode-32k-r2.json`): stream client
disconnects 5 s after first streamed output on a 32K/512 request
(first output 154.1 s — the exact #663 shape). Probes sent at
+5/+15/+30/+60/+90/+119 s **after the close were all admitted with
0.5 s first output** — the request's admission slot, KV slot, and
generation group were released within ≤5 s of the disconnect (the
scheduler's write-failure path: the next `send_event` raises
BrokenPipeError, the generator closes, `cancelled` is set, `_emit`
drops the request, the slot is freed). The allocator held steady at
the warmup baseline through the whole monitor window.

The #663 "zombie" was a measurement artifact: its probes were single
attempts fired inside the same short detection window (and at batch 2,
queued behind the still-active survivor — 503 was correct admission
behavior, not a stuck slot). `urlopen`'s header-admission semantics also
meant the #663 "mid-prefill" cell actually disconnected mid-decode.

## Correction 2 — the OOM warning storm is pre-existing at 32K near-capacity, not disconnect-induced

The growing-allocation OOM warnings (228 MB → 770 MB, ~10 s apart)
appear in **the passing cells' serve logs too**: C1 gate5 (12 warnings),
C2 (16), reuse-identical (12) — byte-identical sizes, deterministic.
They occur around the packed-prefill→decode transition of the FIRST
32K request after warmup on a near-full card (44.5 GiB card, 24 GB
resident weights+KV, ~0.4 GB free at peak): the allocator's retry
spans a transient scratch peak, every attempt fails, the retry with
cache-flush eventually succeeds or the kernel proceeds — requests were
served throughout (all probes 0.5 s). Not a disconnect defect; it is a
known-capacity finding for near-full 32K operation, recorded for the
#60 capacity envelope work.

## The real defect — /v1/models/status was unavailable during any active generation (FIXED)

D3 (`d3-status.json`, pre-fix): polling `/v1/models/status` every 2 s
through one full 32K generation + 45 s decode returned
**99/100 × HTTP 503**.

Root cause: the endpoint routed through
`scheduler._submit` — a queued worker command. Two effects:
1. It competed for request-queue admission — with queue-size 1 and the
   active request holding the only slot, status failed **immediately**
   (BusyError → 503).
2. Even with a free slot, the GPU worker cannot run commands inside an
   active generation group; the bounded 1 s wait expired → 503 for the
   entire generation span.

**Fix (PR: `scheduler.manage('status')` now serves
`Runtime._status()` directly):** the snapshot takes only the
registry lock — no worker, no queue, no admission — and is safe to run
during an active generation (mutations keep the ordered worker path).
Post-fix D3 (`d3-status-fixed.json`): **100/100 polls return 200**
with live allocator telemetry visible mid-decode (21.71 → 21.74 GB as
the KV slot grows). This is what un-blinded the monitoring that
produced the original #663 misdiagnosis.

## Verdict for #664

- Zombie slot + batch-1 "wedge": measurement artifacts, corrected here
  and on #68. Disconnect handling is sound (≤5 s detection+release at
  both batch 1 and batch 2).
- Real defect fixed: status observability during active generation.
- Pre-existing OOM-warning storm at 32K near-capacity: recorded, not
  fixed here (capacity envelope work, #60).

Evidence: this directory (D1 4K + 32K, D3 pre/post-fix, guards, serve
logs). Reproducer: `scripts/experiments/hybrid_q4_disconnect_diagnostic.py`.

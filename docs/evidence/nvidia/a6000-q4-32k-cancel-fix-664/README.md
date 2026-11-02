# A6000 Q4 32K cancel-decode — FIXED, all gates pass (#664, 2026-11-01)

**Result: PASS.** Client disconnect on the Q4 32K executor now cancels the
request within ~one decode step — the zombie slot hold (measured ~160 s
decode-to-completion, and October's 346 s+ window) is eliminated. Guarded
run exit 0, 544 s.

## The defect (two layers, both root-caused with live evidence)

1. **Server: no FIN detection.** A graceful client close sends FIN, not
   RST; writes into a half-closed socket succeed and a small SSE stream
   fits the send buffer, so no write error ever fired and the GPU worker
   decoded the entire request for a client that was gone. Live py-spy
   showed the worker mid-forward 130+ s post-disconnect;
   `admission_outstanding: 1` the whole time.
2. **Harness: urllib's response.close() doesn't close the socket.** The
   socket survives in a reference cycle; the FIN waits for cyclic GC,
   indefinitely late on a quiet harness. The first fixed-server re-run
   still measured a zombie because the *test client* never actually
   disconnected at the TCP level — isolated with a local minimal
   reproduction before re-running.

## The fixes (branch disconnect-detection)

- `server.py`: `peer_gone()` — `select(0)` readability then `MSG_PEEK`,
  checked between generation events; EOF → return, closing the generator
  (`cancelled` → GPU worker drops the request at the next `_emit`).
  Detection bounded by one decode step. Portability: combined
  `MSG_PEEK|MSG_DONTWAIT` stalls on some stacks (macOS: 30 s block →
  TimeoutError, an OSError, misread as gone — every stream request
  aborted; found by tests before it shipped).
- `scripts/experiments/http_disconnect.py`: `hard_disconnect(response)` —
  chase the socket through the response's file chain, close it
  explicitly, force a cyclic collect as fallback. Applied to the Q4
  lifecycle runner and the 35B lifecycle runner (same urllib pattern).
- Retry-bounded release probe in the Q4 runner (the 35B runner's proven
  pattern): an instant probe races the ~one-decode-step release window and
  measures the race, not the hold.

## Measured (run 3, pinned workload: 32K/512, batch 1, chunk 127, disconnect 5 s after first output)

- `early_release_probe`: **admitted at 1.9 s** post-disconnect (previously
  503 until the 160 s zombie completed).
- `cancelled_kv_released: true` — cuda allocator back to baseline
  (21.711 GB vs 21.709 baseline).
- `followup_within_gates: true` — 32K/64 request passed normally after
  release: 155.2 s first output, 64/64 completion tokens.
- Guard exit 0; 12 standing prefill OOM-retry warnings (unchanged,
  separate capacity finding: present in every October cell too — eager
  prefill attention workspace growth at 32K, tolerated via allocator
  retry).

## Tests

76 server tests + 226 subtests, 3 consecutive green runs, including:
- raw-socket close cancels generation before the terminal event
- hard-closed urllib connection cancels identically

## Artifacts

| File | sha256 |
| --- | --- |
| gate-cd3.json | e6d8e3bb545d19715282ba66529c9d94a0f6bc5013abc8ae998199efd79ad3b3 |
| gate-cd3.serve.log | 32bd662544f96937757c3f84ec664b229e3fae78f1b89f6fb2cf84a134019141 |
| guard-cd3.json | c9952b982fd3a36bab136cf15a365e7efd5d069b77f823824ff60eb893555fd0 |

Failed-run findings retained in #664 comments: the raw diagnostic (zombie
mechanism + py-spy evidence), fix-v1 run (harness socket cycle),
probe-race run (1.2 ms probe vs ~3 s release window).

# A30 hybrid placement measurement (#497, parent #64)

Pinned checkpoint: Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
(the #23 identity), loaded from the staged HF snapshot with on-the-fly Q4
packing, context 512, `--no-graph`, compensated precision, `--batch-size 4`,
host budget 40 GiB. Both runs under `guard_shared_cuda` on the A30
(GPU-70f6a991, idle); the harness drives four fixed raw prompts with
`temperature=0`, `max_tokens=32`, two rounds per batch, best round kept.

## Hybrid (CPU/RAM-streamed) placement — completed

`hybrid-run8.json` + `hybrid-guard8.json`: activation warmup 591 seconds
(27B Q4 packing plus streaming setup, inside the first request), then
**3.24 tokens/s aggregate at batch 1** and **6.56 tokens/s at batch 4**
(2.0x concurrency scaling, raw wall-clock over 32-token completions). The
guard completed within its 1800-second bound; temperature and memory gates
held throughout. Earlier failed attempts are retained on the host
(`/tmp/damendaemon-498/hybrid-run{2,3,5}.json`): two 503 bursts traced to a
harness bug (the batch round sent `PROMPTS*batch` — 16 concurrent requests
against queue capacity 8; the live 503-instant snapshot showed 7 queued +
1 owned with the admission semaphore full, i.e. the scheduler's bounded
admission worked exactly as designed), fixed in the harness before this run.

## Resident (fully GPU-resident) placement — not an admitted configuration

`resident-run1.json` + `resident-guard1.json`: the same serve command without
`DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4=1` is rejected at the first request with
HTTP 400 — Q4 execution requires the explicit experimental hybrid opt-in by
policy (`hybrid_q4_policy.py`), and the alternative resident precisions do not
fit this device: FP32 ≈ 108 GiB and Q8 ≈ 28.5 GiB against 24 GiB total A30
memory. The intended two-placement comparison therefore collapses to a single
admitted placement; this is the finding, not a skipped step.

## Consequences for #64

- The hybrid placement is the only supported placement for this checkpoint on
  the A30; its capacity is bounded by host RAM admission, not VRAM.
- Measured per-user goodput for this workload: ~3.2 tokens/s single-user,
  ~6.6 tokens/s aggregate at four users (~1.6 tokens/s per user) — consistent
  with the retained #66 finding (~1.5 tokens/s/user) on HOMER. The provisional
  20 tokens/s/user target is not met by this placement; #66 remains deferred
  and #59/#54 carry the actual limitation.
- Cross-placement numerical identity is vacuous with one admitted placement;
  the numerical gate versus an independent reference was already established
  by #23's A30 84/84 evidence. No performance claim is made beyond these raw
  single-device controls.

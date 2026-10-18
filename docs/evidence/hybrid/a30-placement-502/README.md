# #502 hybrid CUDA stall after the first served batch — root-caused, not a scheduler bug

## Outcome

The reported stall does not reproduce with correct admission. A guarded A30
reproduction (2026-10-17, pinned Qwen3.8-27B snapshot
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, hybrid placement, q4 compensated,
context 512) served two consecutive rounds cleanly:

- batch 1 round: all four prompts served (2.05 tok/s best aggregate)
- batch 4 round: all four prompts served concurrently (4.02 tok/s best
  aggregate), finite greedy outputs identical to the batch-1 round
- guard phase `completed` in 583.8 s, no thermal/reserve violations

The #497 re-run after the fix also served both rounds (3.24/6.56 tok/s,
`docs/evidence/hybrid/a30-placement-497/`).

## Root cause

The comparison harness sent `PROMPTS * batch` concurrent requests — 16 at
batch 4 — against the bounded queue (capacity 8, admission semaphore 8).
Eight requests were rejected immediately with `BusyError('request queue is
full')` → HTTP 503, and the round recorded failures while admitted work
continued. This matches the retained stall evidence
(`docs/evidence/hybrid/a30-placement-hybrid/`: batch-1 round fine, batch-4
round recording HTTP 503s). The fix is in the harness, not the scheduler:
batch rounds send one prompt per batch slot (PR #510). The scheduler-side
admission accounting asymmetry found while diagnosing this (streaming-path
admits uncounted, `admitted_total: 0, released_total: 17`) was fixed in PR
#505 with `admitted_total`/`released_total`/`admission_outstanding`/
`admission_semaphore` in scheduler status.

The issue body's "worker never picks up new work" hypothesis was not
confirmed: with the admission semaphore correctly sized relative to offered
concurrency, the worker admits and serves every round. Rejections are
immediate bounded `BusyError`s, not silent stalls.

## Files

- `run.json` — reproduction report (`completed: true`, both rounds)
- `guard.json` — guard report (phase `completed`, 583.8 s)
- `run.serve.log` — serve log

Measured numbers are raw single-run controls on one host state; batch-4
aggregate varies between runs (4.02 here vs 6.56 in the #497 re-run).

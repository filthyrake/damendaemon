# Qwen3.6-35B-A3B (qwen3_5_moe) Q8_0 32K HTTP lifecycle cells — ALL PASS (#668 cell 3, 2026-11-01)

The MoE's first **production-serve** qualification: `damendaemon serve`
(no probe wrapper — q8_0 fp16 needs no context rescope) with the
`make_executor` MoE dispatch (this PR) and the corrected MoE weight
estimate, on the shared A6000 (pinned UUID, idle at launch), same pinned
snapshot/quantization as the cell-2 gate (PR #670) and C-series
concurrency cells (PR #671): `995ad96e…`, q8_0 packed experts, context
32768, whole prefill (chunked prefill not admitted for this config),
`--no-graph`, `--host-gib 40`, batch/queue per cell.

Runner `scripts/experiments/qwen36_35b_q8_lifecycle_668.py` — adapts the
qualified #68 lifecycle runner; same corpus and pinned slot-0 digest
`31836e96…be727` (the 35B tokenizer reproduces the #68 corpus ids
byte-for-byte). One bounded cell per invocation, guarded each time
(thermal 90 °C, reserve 0, 1800 s; all four guards completed exit 0,
thermal peaks ≤ 74 °C). Per-request gates: first output ≤ 600 s, request
≤ 1200 s, exact token counts. First request per server (~122 s) is
request-driven executor construction (weights 34.76 GiB, load ~115 s) —
admission is lazy by design; each cell warms up before measuring.

## Results — all four cells PASS

| cell | shape | key numbers | verdict |
|---|---|---|---|
| reuse-identical | 32K/512 × 2 | both 512/512; **byte-identical outputs**; no allocator growth; repeat ratio 1.01 (re-prefills, no prefix-cache claim) | **PASS** |
| reuse-suffix | P0, then P_tail × 2 | suffix change → different digest (no stale replay); P_tail repeats byte-identical | **PASS** |
| cancel-decode | 32K/512, disconnect 5 s after first output | disconnect 13.0 s (mid-decode), 114 chars delivered; probe admitted at ~3 s (attempt 2); follow-up 32K/64 first output 7.8 s | **PASS** |
| cancel-prefill (batch 2) | two concurrent 32K/512, slot 0 disconnects 90 s after header admission | survivor 512/512 in 181.6 s (first output 7.75 s); probe admitted at ~3 s **while the survivor still decoded**; KV released to baseline; follow-up clean | **PASS** |

Output digests: the slot-0 32K/512 greedy completion `3a02b8ce…` is
byte-identical across reuse-identical (both requests) and reuse-suffix
round 1 (same pinned prompt), and the 512-token warmup digest
`fe0663fd…` is identical across all four cells — consistent greedy
behavior. The cancel cells use distinct prompt rotations (survivor
slot-1 completion `15267f82…`); the cancelled slot-0 stream in
cancel-prefill delivered 0 chars before the 90 s disconnect (digest =
empty). Note on the C-series slot-0 digest `cc380b42…` (PR #671): that
digest is over the **token-id array** (the executor-direct runner
hashes `json.dumps(token_ids)`), while these serve cells hash the
**decoded text** — the serializations are not comparable, and no
cross-path byte-equality is claimed or implied by either value.

## Finding: the first cancel-decode "failure" was a probe-timing race, not a defect

The first cancel-decode run failed its `early_release_probe` check: the
probe got an **instant HTTP 503** and the follow-up 503'd too — while
the allocator had already returned to baseline and `/v1/models/status`
responded. Diagnosis (guarded diagnostic `diag503c/diag503d.json`,
staged /tmp runner per house rules, never committed): scheduler state
at the failure instant showed `admission_outstanding: 1,
admission_semaphore: 0` — the disconnected request still held admission
**inside the disconnect-detection window**. Measured recovery: the very
next probe at **+3.0 s** admitted normally (first output 1.5 s). The
#68 runner's probe fires ~50 ms after the client close, which never
mattered on the 27B (its cancel-decode cell was a genuine #664 wedge —
never released, status endpoint dead); on the MoE the disconnect
detection works and releases in ~3 s, so the immediate probe races the
release. The runner's `release_probe` now retries (bounded ~10 s,
attempts recorded — a zombie still fails by exhausting attempts) and
the cell re-ran clean. Both artifact sets retained:
`cancel-decode.json` (race finding) and `cancel-decode2.json` (pass).

This is the opposite failure mode from #68's 27B cell 3 — there the
server wedged (zombie request, no recovery in 346 s); here the server
releases correctly and the harness was measuring inside the detection
window.

## Code changes in this PR

1. `make_executor` dispatches `qwen3_5_moe_text` through
   `HybridExecutor` (the serve path's only gap — loader/executor were
   complete from cells 1–2); a MoE model carrying vision weights is
   refused explicitly (the transcribed contract excludes visual tensors
   — a loading defect, not a vision cell).
2. `weight_bytes` (the host pre-load budget estimate) priced the MoE's
   3-D expert stacks at full BF16 (~3× inflation) and would have
   discounted the never-packed routers; it now mirrors
   `quantization.prepare` exactly (3-D stacks packed rowwise per
   expert, routers/combine-gate full precision). Verified against the
   real snapshot: estimate 34.76 GiB == measured `model.bytes`
   34.76 GiB (byte-exact).

## Honest limits

- Lifecycle behavior only (cancellation, reuse, release): no quality,
  SLO, or throughput claim — shared-card wall-clock per repo discipline.
- urlopen caveat inherited from #68: headers arrive at first output,
  so `cancel_after` counts from header admission; both cancel cells
  measured **mid-decode** disconnects (disclosed — a true mid-prefill
  disconnect needs a raw-socket client; at 7.8 s/slot prefill the
  mid-prefill window is small on this model).
- Two early launch attempts of the diagnostic crashed on a harness
  bug (log-helper argument collision) before any GPU work; the clean
  diagnostic is the retained evidence.
- The reuse repeat does not accelerate (ratio 1.01): the hybrid
  executor has no prefix cache — recorded, not claimed as a defect.

## Artifacts (sha256)

| file | sha256 |
|---|---|
| reuse-identical2.json | `8532d5418cdb0375b1b7674bf1ee5d24d8dc2849060ee509dee8ffa40ed3abc7` |
| reuse-suffix.json | `5aea20032d045402afe7fadd2117cda71f6dcd26f613976221103da866e58224` |
| cancel-decode2.json | `52c920c0934a58457c392800d2352a415322d9d157b6ae50026c195b06fbbb73` |
| cancel-prefill.json | `41100990850cd6db2929d1be2323189e80ec4878ed7bb0a8cb134998b44b0d09` |
| cancel-decode.json (race finding) | `489c67b3e700de40669d0c9b32b0fc6554b1b847f4126217fc33c599a0db0547` |
| diag503c.json (scheduler-at-failure) | `e06058979f138f34d56e563d664f1a48331f4d83f415b3d89562bb166c3b61e1` |
| diag503d.json (recovery +3.0 s) | `0cbb7c7f56d4a23f5a24e72cfbe0d478a280f8eb9145fa05cb79c5057f8ab411` |
| guard-reuse-ident2.json | `71e808baecbf9fad12a8dc92e02680926c0b262b1908574f31abd0b35d673828` |
| guard-reuse-suffix.json | `652d3a08ece798c5d01e6d949dc21b8d55172901f60074af4cf4edc311af5b3a` |
| guard-cancel-decode2.json | `d16e14290a70f5e91d4e6865dac24960483df0e29624cc3a1af058b0bd107d6c` |
| guard-cancel-prefill.json | `9b750e45515220d7a3b4931d979230e4aa449bd3508cc38907e54b7014567d60` |

Serve logs alongside each report (mostly quiet — the runner is
incremental-JSON-driven). Diagnostic runner: staged /tmp copy only,
never committed (house rule); its behavior is fully described above.

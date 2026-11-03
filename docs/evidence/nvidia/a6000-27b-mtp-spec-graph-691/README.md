# #691 — Spec-round CUDA graphs vs the eager-MTP constraint (27B, A6000)

**Question.** The MTP draft head refused CUDA graphs at construction — #684's
spec serve ran eager (54.8 s decode window / 512 tokens at C1, on record in
`a6000-27b-mtp-spec-serving-68/`). Can the spec round's main `[B, 2]`
forward be graph-captured with the MTP proposal staying eager between
replays, and does it beat the eager decode baseline with byte-exact output?

**Answer.** Yes — the constraint is lifted and the A/B passes all three
pre-declared gates, but the C1/32K win is modest: **48.3 s graph vs 50.2 s
eager (3.8%, 10.6 vs 10.2 tok/s), byte-identical digests**. The spec round
at 32K context is GPU-bound (the attention GEMMs over the full context
dominate); the graph removes launch+python overhead (~7.4 ms/round), not
GPU work. The structural deliverable is the coexistence: graphs and the
MTP head now verify byte-exactly on the same serve, opt-in, with the
production compensated-surface policy unchanged.

## Cell

| | value |
|---|---|
| Model | Qwen3.8-27B `1d4bf0f2` Q4_0 compensated fp32 compute |
| Device | A6000 `GPU-4adb9ddb-…` (guarded, isolated, ≤90 °C) |
| Shape | C1 — single 32K request, 32,256-token pinned prompt (`31836e96…be727`), 512 greedy tokens, chunk 1024 |
| Serve | `hybrid_q4_a6000_probe serve` (the qualified C-series path) |
| Runner | `scripts/experiments/hybrid_q4_spec_graph_ab_a6000.py` (worktree `mtp-spec-graph-691`, commits `cb393be3` + `c096cc63`) |

## A/B results (both phases, one guarded invocation each)

| phase | decode window | first output | total | tokens | digest | peak alloc |
|---|---|---|---|---|---|---|
| eager run 1 | 50.1 s | — | — | 512/512 | `db7acaaee335e4e6…` | — |
| eager run 2 | **50.2 s** | 129.7 s | 179.8 s | 512/512 | `db7acaaee335e4e6…` | 21.55 GiB |
| graph run 2 | **48.3 s** | 130.5 s | 178.7 s | 512/512 | `db7acaaee335e4e6…` | 21.57 GiB |

- **EXACTNESS**: graph digest byte-identical to eager (and eager run 1
  reproduced run 2's digest — determinism across full serve restarts).
- **DECODE**: 48.3 < 50.2 s — strictly shorter (gate). Prefill first output
  unchanged (129.7 vs 130.5 s, reported not gated; identical scan prefill).
- Graph pools cost +0.02 GiB peak — negligible against the 21.5 GiB cell.
- Eager control reproduced the #684 record's magnitude (50.1/50.2 vs 54.8 s
  on record; inside the declared 30–80 s control band).

## The design

- `grouped_attention.window(q, k, v, positions)` — static full-context
  grouped attention: `allowed[b, r, c] = key c <= positions[b, r]`; k/v are
  the FULL persistent cache (no `[:end]` slicing); unwritten tail positions
  read as zero keys and are mask-excluded; **no host-value reads inside**
  (shape/dtype checks only — position ownership is `spec_round`'s
  structural guarantee).
- `HybridExecutor.spec_graphs` — the `[B, 2]` main forward captured per
  batch size on the qualified surface (CUDA, Triton fused, fp32 KV,
  compensated, MTP head, `DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH=1`). Replay is a
  pinned metadata copy (tokens, per-row divergent positions, slots) +
  `graph.replay()`; the MTP proposal/backfill stays eager on the replayed
  hidden buffer; rejected-slot rollback restores the same persistent
  buffers the graph mutates.
- Production `hybrid_precision.select` is UNCHANGED — graphs on the
  compensated surface are admitted only in the probe's isolated
  qualification view under the opt-in. Default everywhere remains eager.

## The two live findings

1. **Graph run 1 failed at construction** (retained:
   `graph-run1-failed.log`, `691-graph-run1-failed-guard.json`): the
   pre-existing PLAIN batch decode-graph capture ran
   `forward_embeddings`, whose MTP stash does `slots.tolist()` — a host
   read, uncapturable. Deeper: a replayed plain decode graph would
   silently skip the MTP hidden/KV maintenance (stale draft seeds, MTP KV
   holes). Fix (`c096cc63`): plain decode-graph capture is skipped when
   the head is armed; plain decode steps run eager (the #684-measured
   exact path); the greedy spec serve never calls `decode_batch`.
2. **A window-mask broadcasting bug caught by the CPU parity tests before
   any GPU run**: without an explicit singleton head axis, the
   `[batch, rows, context]` mask right-aligns its batch axis onto the
   repeats axis — silently wrong whenever `batch != repeats`, numerically
   plausible when equal (`heads/kh == batch`). Pinned by
   `tests/test_spec_graph_691.py::GroupedWindowTests`.

## CPU contracts (`tests/test_spec_graph_691.py`, 5 tests)

Window parity vs eager per-row masked SDPA (divergent positions,
unwritten-tail exclusion), window rejection contract, scratch-partition
behavior preservation, `_spec_main_forward` (the captured body) parity vs
the eager forward — logits, hidden, and persistent state — and the env
validation. Full local battery failure-set diff vs `origin/main`:
identical sets (pre-existing host issues, see #697); zero regressions.

## Artifacts

- `eager2.json` / `graph2.json` — phase reports (round records, digests, statuses)
- `eager-run1.log` / `eager-run2.log` / `graph-run1-failed.log` / `graph-run2.log` — runner logs
- `691-eager-run1-guard.json` / `eager-run2-guard.json` / `691-graph-run1-failed-guard.json` / `graph-run2-guard.json` — guard reports
- `eager2.serve.log` / `graph2.serve.log` — serve logs (quiet; milestone prints live in the runner logs)

## Honest decomposition

The 3.8% is launch-overhead removal only (~7.4 ms/round × ~256 rounds); the
attention GEMMs over 32K keys — identical in both phases — dominate the
round. No decomposition beyond that is claimed. Shorter-context shapes,
where the launch share is larger, are follow-up headroom on the now-open
graphs lever; unmeasured, unclaimed here.

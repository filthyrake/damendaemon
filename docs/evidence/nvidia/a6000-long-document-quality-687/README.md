# Long-document quality gates at the qualified long-context tiers (#687)

The one undelivered acceptance bullet from #60's list: capacity and SLO
are measured; quality is not. This battery measures it — judge-free,
deterministic, pinned corpora where every ground truth is known by
construction, so scoring is normalized exact string agreement (never a
judge model; no cross-engine claims). Runner:
`scripts/experiments/long_context_quality_battery_687.py`; CPU tests pin
the scorer and corpus assembler (`tests/test_quality_battery_687.py`, 8
tests). Gates declared on #687 before any run, applied unchanged.

## Tasks

- **T1 positional retrieval with distractors** — 3 code needles at
  ~25/50/80% of the corpus + 6 same-format distractor needles; multi-query
  single request. Gate ≥ 2/3.
- **T2 multi-needle** — 6 station inventory counts spread through the
  corpus. Gate ≥ 4/6.
- **T3 cross-document composition** — 2 shipment × cold-chain-directive
  joins, one YES and one NO (the queried permit is the wrong one).
  Gate ≥ 1/2.
- **T4 summarization** — event log, 5 pinned critical facts in routine
  entries, ≤ 80-word summary; gate reads the response's final 120 words.
  Gate ≥ 3/5.

## Results (A6000, guarded, one invocation per tier/phase — 6 cells)

| tier (production shape) | T1 | T2 | T3 | T4 | verdict |
|---|---|---|---|---|---|
| 35B MoE q8_0 32K (C6 serve) | **3/3** | **6/6** | **2/2** | **5/5** | **qualified, 16/16** |
| 27B dense Q4 32K (C4 spec serve) | **3/3** | **6/6** | 1/2 | **5/5** | **qualified, 15/16** |
| 27B dense Q4 64K (C1 spec serve) | **3/3** | **6/6** | 1/2 | **5/5** | **qualified, 15/16** |

- Retrieval with distractors and multi-needle extraction are **perfect
  across all three tiers** — 30/30 items — including the 27B Q4 at 64K.
- Summarization **5/5 everywhere** (tail-gated; reasoning quotes excluded).
- The single recurring miss: **Q6, the negative cross-document join** (a
  NO answer — the queried permit is the wrong station's), missed by the
  27B at both 32K and 64K; Q5 (the positive join) passed everywhere. The
  35B got both. Characterized and inside the declared 1/2 allowance;
  recorded per-item in the reports.

All 3 tiers pass all 4 gates at both phases → **#687 acceptance met.**

## Retained findings (failures preserved, per the discipline)

1. **qwen27-32k qa run 1 — 0/11 parse failure (`687c.json`/`g687c.json`):**
   the 27B answered in the reasoning format (Q-restatement +
   "Looking through the text: …" + "Answer: X"), retrieving the amber
   needle **verbatim and correctly**, but the declared A1–A6 format never
   appeared and the 256-token budget cut the response off mid-Q2. Task-
   design finding, not a retrieval failure. Amendment declared on #687
   before any re-run: scorer accepts both formats truth-blind (final
   `Answer:` line per Q-segment), summary gate reads the response tail,
   decode budgets raised (qa 700, summary 384) — gate VALUES unchanged.
   The failed run stands in the record; the 35B's 16/16 (original
   scorer) is unaffected and passes both scorers.
2. **Two launcher iterations that never reached measurement** (disclosed,
   not retained as reports): a corpus-budget overflow refusal (prompt
   32712 > 32768−576) and an HTTP 400 (prompt + max_tokens > context
   after the budget raise) — both fixed in the runner (retry-with-
   shrink corpus assembly; budget derived from the decode budget:
   prompt + max_tokens + 64 ≤ context). Their guard JSONs were overwritten
   by the successful relaunches into the same paths; the tracebacks are
   quoted in the transcript.
3. The Q6 negative-join miss is consistent across depths for the 27B —
   a model-behavior characterization (quantized Q4 at both tiers, same
   prompt text), not a harness artifact; the item passes on the 35B.

Honest scope: these gates qualify **extraction/composition/summarization
against constructed ground truth** — they do not measure prose quality,
and needle positions, while spread ~15%–96%, are deterministic (fractions
recorded per run). Timings are recorded, never gated (latency is the SLO
lanes' deliverable): the 35B cells complete in ~10.5 s request time; the
27B 32K ~200 s; the 27B 64K ~412 s — dominated by prefill, consistent
with the SLO cells' records.

## Artifacts

- `qwen35-32k-qa.json` / `qwen35-32k-summary.json` (+ guards) — 16/16
- `qwen27-32k-qa-failed-parse.json` / `g687c.json` — the run-1 finding
- `qwen27-32k-qa.json` / `qwen27-32k-summary.json` — 15/16
- `qwen27-64k-qa.json` / `qwen27-64k-summary.json` — 15/16
- `runner.py` — the battery verbatim; `test_quality_battery.py` — the
  CPU scorer/assembler contract

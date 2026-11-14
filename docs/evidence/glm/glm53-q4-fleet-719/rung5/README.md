# GLM-5.3 q4 fleet execution — rung 5: the 4-host full-model chain (#719)

2026-11-09/14, guarded cells on labnode3 (A30) / rocky10-desktop (A6000) /
3090box (RTX 3090) / homer (8060S — pending). Stage cut at full-indexer
boundaries [[0,34),[34,50),[50,66),[66,78)]; intra-host hops are FILE
handoffs (dump → next cell `--inbound-file`), cross-host hops scp the dump —
no TCP after the startup-race findings.

## Status: 66 of 78 layers executed, ALL gates green

| cell | host | layers | parity gates | worst rel RMS |
|---|---|---|---|---|
| 1a | labnode3 A30 | 0–13 | 11/11 PASS | 0.1027 |
| 1b | labnode3 A30 | 14–25 | 11/11 PASS | (see json) |
| 1c | labnode3 A30 | 26–33 | 8/8 PASS (layers 0–2 dense) | (see json) |
| 2a | rocky10 A6000 | 34–41 | 8/8 PASS | 0.1143 |
| 2b | rocky10 A6000 | 42–49 | 8/8 PASS | 0.1152 |
| 3a | 3090box 3090 | 50–57 | 8/8 PASS | 0.1178 |
| 3b | 3090box 3090 | 58–65 | 8/8 PASS | 0.1331 |

Every MoE layer's paired-fp8-oracle parity gate (rel RMS ≤ 0.15 AND
max_abs ≤ 0.15·absmax, on the ACTUAL chain activations, same fp32 router
selection, same accumulation order) passed. Chain state crossed vendors
A30 → A6000 → 3090 via verified file handoffs (md5-checked dumps).

**Stage 4 (homer, layers 66–77 + final norm + lm_head): pending host
availability.** Its exact inbound — `handoff-layer66.pt` — sits on
3090box; the rung completes with one guarded cell + stewardship whenever
homer frees up.

## The honest engineering record (attempts 1–8, all retained in #719 comments)

The model path never failed: parity numbers were identical across every
attempt (deterministic chain). All failures were launch plumbing, each
found and fixed: guard tool ceilings (1800 s) → chunked cells; seed device
placement; shared-indexer layers shipping no indexer weights; TCP
handoff startup race (3×) → file handoffs + dump-then-connect; trigger
grep patterns; inbound-file cells binding listeners; one sequencer
version that dropped `--stage` from launches (caught by init-verification,
zero GPU time lost). Pack rates measured per host: A30 106.6 / A6000
102.1 / 3090 106.1 / 8060S 70.7 s/layer — the APU packs fastest (no PCIe
crossing for the host-resident banks).

Banks: 75 q4_0 layers persisted fleet-side (~310 GiB; 3090box banks on
/home — its /tmp is a 70 GB root slice while /home is 7.2 TB). Attention
and indexers run shipped-fp8 everywhere (rung-4b decision: selection
exactness #668 — q4 attention perturbs the DSA indexer via the MLA
low-rank, 23.8% selection divergence measured).

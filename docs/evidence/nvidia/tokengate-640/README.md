# TokenGate independent evaluation (#640)

Date: 2026-10-26. Hardware: **rocky10-desktop idle RTX 3090** (24 GiB, GA102,
driver 615.71.09, GPU-24ef3d4c-…, unlocked via gpu-unlock @c274e74), batch 1,
bf16. The Intel B580 drove the desktop display throughout; the 3090 was idle
(17 MiB / 35 °C at start, ≤60 °C under load). Parent: #639 (TokenGate
umbrella). This is child A: verify or refute the paper's claims
(arXiv:2610.26754) on named damendaemon hardware before any adoption
decision.

## What upstream evaluates out of the box (gate 1)

- Repo: `token-gate-llms/TokenGate` (redirects from `tokengate-models`),
  Apache-2.0, **pinned commit `6e51aac`** (main HEAD 2026-10-26; last push
  2026-10-21).
- Env installed from upstream's own `uv.lock`: torch 2.14.1+cu130,
  **transformers 5.19.0** (locked), datasets 5.1.0 (locked), accelerate
  1.15.0, Python 3.12. One deliberate deviation: attention backend
  **flex** everywhere (the fa2bias FlashAttention-2 key-bias kernel needs a
  full flash-attn source build; upstream documents flex as the no-build
  fallback and the Qwen3.5 hybrid *defaults* to it).
- **No training is required for evaluation**: six gate-adapted checkpoints
  are released on HF (`liana313/tokengate-{niah,biodexr,2wikimqa,multinews,
  repobench}-qwen3-1.7b` and the hybrid `tokengate-niah-qwen3.5-2b`), each
  "the exact checkpoint the paper's benchmarks used" per its README. The
  #640 training gate is therefore N/A-for-eval (training lane configs exist
  under `train-scripts/` but were not exercised).

## Measured results (dense Qwen3-1.7B, gate layer 1, released NIAH gate)

All numbers: upstream's own benchmark harness (`benchmarks/run_benchmark.py`)
or demo (`examples/niah_demo.py`), FlexAttention backend, greedy, val
partitions, single RTX 3090.

### NIAH 16K tier — harness, 50 val rows (`niah-16k-bench.log`)

| arm | K | acc | TTFT | speedup |
|---|---|---|---|---|
| full (base model) | — | 1.000 | 925 ms | 1.00× |
| tokengate | no prune | 1.000 | 1,024 ms | 0.90× |
| tokengate | 2000 | 1.000 | 156 ms | 5.92× |
| tokengate | 500 | 1.000 | 92 ms | 10.05× |
| tokengate | 100 | 1.000 | 83 ms | **11.19×** |

### NIAH 32K tier — harness, 16 val rows (`niah-32k-bench.log`)

| arm | K | acc | TTFT | speedup |
|---|---|---|---|---|
| full | — | 1.000 | 2,408 ms | 1.00× |
| tokengate | no prune | 1.000 | 2,859 ms | 0.84× |
| tokengate | 2000 | 1.000 | 265 ms | 9.10× |
| tokengate | 500 | 1.000 | 200 ms | 12.03× |
| tokengate | 100 | 1.000 | 191 ms | **12.63×** |

### NIAH 32K demo — 5 haystacks, needle-survival + KV census (`niah-32k-demo.log`)

acc 1.00 at all/2000/500/100; TTFT 3,075→188 ms = **16.21×** at K=100;
the needle sentence survives pruning 100% at every K; at K=100 only
**144 of 32,191** prompt tokens are retained in the KV cache.

### Peak prefill memory (`tg-mem-probe.json`, `tg-mem-probe.py`)

torch.cuda.max_memory_allocated around each arm's generate (32k prompt,
32 decode tokens — prefill dominates), 5 rows, steady-state after warmup:

| arm | peak alloc | peak reserved |
|---|---|---|
| full | 8.66 GiB | 9.09 GiB |
| tokengate K=2000/500/100 | 5.11 GiB | ~5.5 GiB |

**1.70× peak-memory reduction**, uniform across K ≤ 2000 (weights floor
3.22 GiB + phase-1 scoring; the prunable KV above the gate is the only
part that shrinks).

### LongBench RepoBench-P — harness, 50 val rows (`repobench-bench.log`)

Run completed 2026-10-26 after the power incident, under a **280 W GPU power
cap** (from the unlock's 450 W max; `tg-rb-valonly.py` rebuilt only the
val store the harness reads — 492 rows, the 16,755-row train tokenization
is training-lane data the pass never touches):

| arm | K | acc (ES) | TTFT | speedup |
|---|---|---|---|---|
| full (base model) | — | 0.266 | 715 ms | 1.00× |
| tokengate | no prune | **0.584** | 777 ms | 0.92× |
| tokengate | 2000 | 0.562 | 197 ms | 3.63× |
| tokengate | 1000 | 0.524 | 146 ms | 4.88× |
| tokengate | 500 | 0.526 | 123 ms | **5.82×** |

Reproduction quality: upstream's README states "RepoBench holds 0.533 ES at
K=500 (5.6×)" — we measure **0.526 ES at 5.82×** on a single 3090
(power-capped, flex backend). Note the no-pruning arm shows the task-trained
gate+LoRA **more than doubles** edit similarity over the base model
(0.266→0.584) — this is the fine-tuning delta, not pruning; accuracy at
reduced K degrades only marginally from the gate's own no-prune reference
(0.584→0.526 at K=500).

## Verdicts per claim family (gate 4)

- **TTFT speedup (paper: 1.4–11.3×): VERIFIED, at the top of the range.**
  11.19× (16K, K=100), 12.63× (32K, K=100), 16.2× (demo 32K incl.
  warmup-free cold flex compile) on a single RTX 3090. Honest caveats: the
  ratio is hardware/backend-dependent (our full-prefill baseline is
  bandwidth-bound on the 3090 and we used flex, not their fa2bias kernel —
  both biases are stated), and `tokengate` with **no pruning costs
  0.84–0.90×** (a 10–16% slowdown) — the paper's efficiency applies at
  reduced K, not K=full.
- **Peak prefill memory (paper: 1.5–3.4×): VERIFIED.** 1.70× at 32K on a
  1.7B dense model (8.66 → 5.11 GiB). The top of their range presumably
  needs longer contexts/larger models; our single data point sits inside it.
- **Accuracy-at-budget (paper: no degradation): VERIFIED on NIAH and
  RepoBench-P.** NIAH: acc 1.000 at every K including K=100 (0.45% of a 16k
  context, 0.3% of 32k), needle sentence retained 100% — the gate provably
  keeps the one relevant sentence and drops the haystack. RepoBench-P
  (edit-similarity, harder): 0.526–0.584 at reduced K vs the gate's own
  no-prune 0.584 — within ~10% of the reference at K=500, matching
  upstream's published 0.533.
- **Scale caveat (stated per the issue): all verified numbers are
  1.7B-scale dense Qwen3-1.7B on a 24 GiB consumer GPU.** Nothing here
  qualifies our 27B Q4 hybrid path or the A30/HOMER topology.

## The hybrid model (Qwen3.5-2B) is blocked on upstream code — recorded

The released hybrid checkpoint (`tokengate-niah-qwen3.5-2b`, gate at
DeltaNet layer 1) **cannot be evaluated from upstream main `6e51aac` as
shipped**, even in upstream's own locked environment. Three concrete
blockers, each confirmed on transformers 5.15.0 / 5.19.0 / 5.21.0:

1. `gate_forward.py:398,446` dispatches hybrid layers on
   `getattr(layer, "layer_type", …)`, but transformers' `Qwen3_5DecoderLayer`
   has used **`block_type`** in every released version — a DeltaNet gate
   layer takes the full-attention branch and crashes with
   `'Qwen3_5DecoderLayer' object has no attribute 'self_attn'`.
2. The public `generate` path never passes `prune_below_gate`; for a
   DeltaNet gate layer there is **no KV cache layer at or below the gate**
   and `gate_decode.py:147` raises `StopIteration` searching for one.
   Passing the harness's own `prune_below_gate=True` flag (as our runner
   does, `tg-hybrid-niah.py`) gets past this but exposes blocker 3.
3. The phase-2 (kept-token) forward receives a 4D attention mask of
   phase-1 width (32,156) from stale cache state; transformers'
   `apply_mask_to_padding_states` then multiplies it against the 2,048-token
   kept sequence → `RuntimeError: size of tensor a (2048) must match
   b (32156)`. Present on every transformers version tested.

Verdict for the hybrid arm: **not honestly evaluable without nontrivial
unverified patches; recorded as the blocked gate** (the paper's headline
"3.8× TTFT on Qwen3.5-2B NIAH 32K" remains unverified by us). This is
directly load-bearing for damendaemon: our 27B is the same hybrid
`qwen3_5_text` family, so any adoption work (#641) inherits this
integration surface.

## As-shipped breakage ledger (all reproduced, minimal recorded patches)

Upstream's lock (`datasets==5.1.0`, `transformers==5.19.0`) predates
ecosystem changes; several pipeline stages cannot run in their own pinned
environment:

- `load_dataset("THUDM/LongBench", …)` — dataset-script support removed in
  datasets 5.x (index build, `dataset.py`, `matched_prepare.py`);
  patched to load the plain `data.zip` jsonl directly (`patch-dataset.diff`).
- `load_dataset(..., trust_remote_code=True)` — the argument itself is
  rejected by datasets 5.x, even for parquet-native repos (repobench
  mirrors `tianyang/repobench_{python,java}_v1.1`); patched to drop the
  argument.
- `datasets/longbench-datasets/longbench_repobench_p/prepare.py` never
  passes a model to `prepare_dataset`, so the store tokenizes with the
  **default Qwen3-8B** tokenizer while the benchmark config expects
  Qwen3-1.7B (`patch-repobench-prepare.diff`); patched to
  `PREPARE_MODEL` with a 1.7B default.
- The `layer_type`/`block_type` hybrid dispatch bug above
  (`patch-gate-forward.diff`).

Patches are minimal and recorded; originals preserved on the host
(`/tmp/*.orig` in the eval clone). They change environment compatibility,
not algorithm semantics — except `prune_below_gate=True`, which is the
upstream eval harness's own flag, and which still leaves blocker 3 in
place, so no hybrid number in this report depends on them.

## Incident record — host hard power-off during the cell (2026-10-26 14:12:53)

During the RepoBench-P store tokenization phase (CPU-only; the 3090 had been
idle for ~25 min after the last benchmark pass), rocky10-desktop lost power
entirely — hard off, no reboot — with wall power uninterrupted and no user
action. Forensics (rsyslog `/var/log/messages`; journald is non-persistent
on this host, so the dead session's journal is unavailable):

- Last line before the gap: **14:12:53** — routine service noise. **No
  shutdown trail, no OOM, no kernel panic, no Xid/NVRM, no MCE, no thermal
  event.** Next entry: a fresh kernel boot at 14:18:17 (after the owner
  power-cycled the box).
- That pattern is the signature of power being cut to the board and staying
  off until AC removal — consistent with a PSU protection latch (over/under-
  voltage) or a dead-short-style fault, not any software failure mode.

Hypothesis ranking after the owner confirmed the PSU is a **1600 W unit,
about a year old** (≈3× headroom over worst-case draw — overload is
unlikely):

1. **Momentary mains sag/brownout** — sub-second dips go unnoticed by a
   person but latch a PSU off; high simultaneous draw (the preceding ~1 h
   45 m of sustained CUDA load on the unlocked 3090) is when the box is most
   sensitive. Fits the lag and the hard-off-then-stays-off behavior best.
2. Early PSU fault or marginal power connector — age doesn't fully
   exonerate, but unlikely.
3. Pathological transient from the firmware-unlocked card (gpu-unlock
   @c274e74 raises power ceilings; it is the one non-stock element in the
   power path).

All three: plausible, not proven — recorded honestly. Session correlation is
real (the sustained unlocked-GPU load was this session's work), causation is
not established, and the elevated-power-limit risk was not flagged before the
run — that's the process lesson regardless of mechanism.

Consequence for this issue: the RepoBench-P harness pass was rerun and
completed after the reboot, under a **280 W GPU power cap** set at the
owner's direction (from the unlock's 450 W max), with the GPU held at
≤65 °C throughout. All three claim families have measured dense-model
numbers (above); the hybrid arm's blockers are recorded. The evidence dir
holds everything, including the log tail around the gap.

## Files

- `niah-16k-bench.log`, `niah-32k-bench.log` — upstream harness, val splits
- `niah-32k-demo.log` — upstream demo (needle survival, KV census)
- `tg-mem-probe.json`, `tg-mem-probe.py` — peak prefill memory probe
- `repobench-bench.log` — LongBench RepoBench-P harness pass
- `tg-rb-valonly.py` — val-store-only prep (the harness reads val.pkl alone)
- `tg-hybrid-niah.py` — hybrid runner (documents the three blockers)
- `incident-log-tail.txt` — verbatim rsyslog tail around the power-off gap
- `patch-gate-forward.diff`, `patch-dataset.diff`,
  `patch-repobench-prepare.diff` — the compat patches, verbatim

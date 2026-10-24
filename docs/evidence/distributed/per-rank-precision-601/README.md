# Per-rank heterogeneous precision: 27B mixed-format chain cell (#601)

Date: 2026-10-23 · Topology: the #536 gate-4-qualified three-rank,
two-host chain — labnode3 A30 (rank 0, coordinator) + A2 (rank 1), desktop
RTX 3090 (rank 2, output head) · Checkpoint: Qwen3.8-27B `1d4bf0f2` ·
Runners: `scripts/experiments/gate4_mixed_precision_chain.py`,
`scripts/experiments/gate4_mixed_compare.py` (torch 2.13.0+cu130 /
NCCL 2.29.7, identical on both hosts)

## Passes

Two passes over the pinned gate-4 trace (context 2048, batch 1, seed
`range(16,48)`, 12 greedy tokens, standard autoregressive decode), both
with the SAME layer split [0, 24, 44, 64] so both inter-rank boundaries
sit identically:

| Pass | Formats (rank 0/1/2) | Rank 0/1 path | Rank 2 path |
|------|----------------------|---------------|-------------|
| reference | `q4_0,q4_0,q4_0` | compensated Q4 Triton (qualified) | compensated Q4 Triton |
| mixed | `q4_0,q4_0,none` | compensated Q4 Triton (identical) | unquantized FP16 Triton |

Startup agreement: the shared chain contract (`stage_identity.chain_contract`:
config, context, the full per-rank format vector, FP32 boundary dtype) —
agreed by all ranks before execution in both passes.

## Verdict (`comparison.json`, all cross-pass comparisons scoped to the
input-identical rounds 0..9 — after the first greedy divergence the two
passes feed different tokens, so their activations differ by construction
and are excluded, not compared)

- **Boundary transport bit-exact, both streams** (`rank0_to_1`,
  `rank1_to_2`): every FP32 boundary activation digest (SHA-256) identical
  across the passes for all 10 input-identical rounds. Ranks 0/1 execute
  the identical qualified Q4 path in both passes, and identical inputs
  produced bit-identical boundary outputs — the divergence is provably
  isolated inside the mixed head rank.
- **Greedy: 10/12 agreement.** Divergence at index 9 (reference `1`,
  mixed `271`); the streams re-converge at index 11 (`220` both).
- **Head logits on identical boundary inputs** (the mixed head's measured
  divergence from the Q4 head): max_abs **2.62** (round 9),
  max rms_rel **0.121**; per-round argmax agreement 9/10 — every round
  agrees except round 9 itself, where the argmax flips. Round 0 (identical
  argmax `70182`): max_abs 1.51, rms_rel 0.093. Mixed precision across
  ranks does not compose bit-exactly by construction; this is the
  calibrated measurement, reported exactly as measured.

Timings (warmup-inclusive, informational only — not performance claims):
reference prefill 4.63 s, steady decode ~159 ms/step; mixed prefill 4.68 s,
~163 ms/step (the unquantized head is not measurably slower at this
scale). Rank-0 peak allocation 7.42 GiB (both passes, budget 22528 MiB).

## Capacity demonstration

The per-rank-format planner's own split for the same budgets/formats is
`[0, 35, 61, 64]` (`mixed-coordinator.json` `planned_mixed_split`): with
rank 2 unquantized the planner shifts 11 more layers onto the A30 rank
(0–35 vs 0–24) because the 3090's FP16 stage is priced at its true cost.
The run itself pinned the reference split so both passes share
boundaries.

## Reproduction

On both hosts (staged checkout at `/tmp/damendaemon-601`), per pass:

```bash
# labnode3 (node 0, ranks 0-1)
cd /tmp/damendaemon-601 && PYTHONPATH=. GATE4_FORMATS=<q4_0,q4_0,q4_0|q4_0,q4_0,none> \
  /home/damen/dllm-dev/venv/bin/torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \
  --rdzv_backend=static --master_addr=10.30.100.3 --master_port=2950x \
  scripts/experiments/gate4_mixed_precision_chain.py

# desktop (node 1, rank 2; CUDA_VISIBLE_DEVICES pins the 3090 UUID)
cd /tmp/damendaemon-601 && PYTHONPATH=. GATE4_FORMATS=<same> \
  CUDA_VISIBLE_DEVICES=GPU-24ef3d4c-… /home/damen/3090-damendaemon/venv/bin/torchrun \
  --nnodes=2 --node_rank=1 --nproc_per_node=1 --rdzv_backend=static \
  --master_addr=10.30.100.3 --master_port=2950x scripts/experiments/gate4_mixed_precision_chain.py

# verdict (on the rank-0 host with both passes' reports + captures)
/home/damen/dllm-dev/venv/bin/python scripts/experiments/gate4_mixed_compare.py
```

Admission follows the recorded gate-4 probe convention (context
512..32768 and the remote-tail-rank substitutions; every other guard
intact; production admission unchanged).

Determinism: the reference pass was run three times end-to-end
(protocol fix re-run x2); greedy output was byte-identical across all
three, and the mixed pass twice.

## Incident record (part of what this evidence covers)

- The first launch crashed on a runner bug (`prefill` returns the head's
  token row, not a scalar) — fixed (`first[0]`).
- One NCCL-teardown stall after STOP left rank 2's capture writeback
  unexecuted; captures are now flushed per round from the forward hook,
  which recovered the data on the re-run. The stall itself is recorded
  here, not diagnosed further (control-plane teardown, outside #601 scope;
  all rank processes were killed cleanly after each pass).
- An operator error during capture consolidation deleted the desktop's
  first reference-rank2 captures; the reference pass was re-run to
  regenerate them (determinism confirmed across the three reference
  executions).

## Files

- `reference-coordinator.json` / `mixed-coordinator.json` — rank-0
  reports (greedy, timings, planned mixed split, peak allocation).
- `reference-rank1.json`, `mixed-rank1.json` — rank-0→1 boundary digests.
- `reference-rank2.json`, `mixed-rank2.json` — rank-1→2 boundary digests
  + head logit digests/argmax per round.
- `comparison.json` — the verdict record.

## Scope

Bounded qualification cell: two passes, three ranks, two hosts, one
model, greedy agreement, storage-planner pricing. NOT qualified here:
mixed chain HTTP serving (the CLI path is implemented but its serving
admission/SLOs are future work), resident model switching, Q8_0 ranks
(stays rejected with its policy reason), and any performance claim.

# #58 — Real-payload promoted/packed prepare: both 2×96 fleet stages materialize from the official weights

**Third physical-qualification increment** (after the #702 contract audit and
the #703 fleet storage plan): the first #58 cells to read real weight
PAYLOAD bytes — the official `DeepSeek-V4-Flash-0731` checkpoint on the NAS
was loaded into promoted/packed CPU stage residency exactly per the 2×96 GiB
fleet plan, with byte-exact spot verification against independent
`safetensors` reads. No execution, no model output, no GPU.

## Cell A — boundary/mid single-layer stages, payload spot-checks (25/25 PASS)

`load_reference_stage` on three ranges; every check compares the loaded
tensor against a direct `safe_open` read of the same checkpoint tensor:

| stage | blocks | load | spot-checks |
|---|---|---|---|
| [0,1) embed boundary | 1 | 13.0 s | **9/9** (incl. embed.weight fp32-from-bf16 exact) |
| [22,23) mid | 1 | 10.3 s | **7/7** (no tid2eid — hash layers only, layers 0–2) |
| [42,43) head boundary | 1 | 24.7 s | **9/9** (incl. head.weight, norm.weight) |

Verified per stage: fp8 attention payload + E8M0 block scales
(`wq_a.weight`/`wq_a.scale` uint8 byte-exact), fp4 expert payload + scales
(`experts.0.w1`, `experts.255.w2` byte-exact), fp32-converted floating
tensors (`ffn.gate.weight`, `attn_norm.weight`, embed/head/norm —
bf16→fp32 exact), and the I32 hash routing table `ffn.gate.tid2eid`
(values exact). Peak RSS 9.2 GiB; cleanup releases to 0.3–0.4 GiB.

## Cells B/B2 — both full fleet stages of the 2×96 plan

| stage | blocks | load | RSS after load | planner predicted retained | peak RSS |
|---|---|---|---|---|---|
| [0,22) | 22 | 397.1 s | **75.9 GiB** | 75.61 GiB | **79.4 GiB** |
| [22,43) | 21 | 356.6 s | **72.7 GiB** | 72.32 GiB | **76.2 GiB** |

- Measured residency lands within 0.4% of the #703 header-only planner's
  declared retention on both stages — the plan's arithmetic is now
  payload-confirmed, and both peaks sit inside the 96 GiB per-host budget
  of the adopted fleet plan (also inside the tight 2×80 shape).
- Structure exact on both: compress ratios match the config tail
  ([4,128] alternating after two ratio-0 layers), 256 routed experts + 1
  shared expert per block (70.64 GiB packed expert bytes held in [0,22)),
  embedding only on [0,22), head only on [22,43).
- Cleanup hygiene: after `stage.cleanup()` + gc, process RSS returns to
  sub-GiB before exit; system returns to 118 GiB available (labnode3's
  A30 lane untouched throughout — CPU-only, no GPU allocation).

## Honest scope

- Payload READ + reference-format residency only: no forward pass, no
  logits, no generation, no execution admission — `model_execution_qualified`
  remains false for the official checkpoint. The teacher-forced forward
  against real weights is the next cell.
- Loads ran on labnode3 (118 GiB available) from the read-only NAS mount;
  nothing was written to the NAS; scratch only under `/tmp`.
- Environment: `/home/damen/dsv4f-venv` (python3.11, CPU torch 2.15,
  safetensors, tokenizers); staged repo tree at `/tmp/damendaemon-58`
  at `c3cd86f4`.

## Artifacts

- `cell-a.log` / `dsv4f_payload_cell_a.py` — 25/25 spot-check cell
- `cell-b.log` / `dsv4f_payload_cell_b.py` — [0,22) prepare cell
- `cell-b2.log` / `dsv4f_payload_cell_b2.py` — [22,43) prepare cell

## Next under #58

Teacher-forced single-stage forward + logits/routing/compression-state
parity against the reference on real weights (CPU, bounded), then the
two-process host topology on the 2×96 plan, then quality/serving tiers.

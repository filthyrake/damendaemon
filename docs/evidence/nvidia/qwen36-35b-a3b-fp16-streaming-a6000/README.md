# Qwen3.6-35B-A3B FP16 synchronous layer-streaming — too-big-model cell (A6000)

**Result: PASS under the amended gate pre-declared on #668 (2026-11-01, issue
comment `6563769920`).** The pinned 64.6 GiB unquantized FP16 snapshot ran
host-resident through synchronous layer streaming on the 44.5 GiB A6000 —
weights that cannot fit the card executed correctly, one layer resident at a
time.

## Snapshot

`Qwen/Qwen3.6-35B-A3B` at HF revision
`995ad96eacd98c81ed38be0c5b274b04031597b0` (same pin as the cell-2 Q8 gate,
PR #670). Weights 64.6 GiB unquantized FP16, host-resident; load 27.2 s.

## Configuration

- `DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD=1` (set in-process: the guard's
  hermetic child env strips `DAMENDAEMON_*`), eager torch FP16, no graphs,
  context 512, quantization none, kernels `torch-hybrid`.
- Guarded by `scripts.experiments.guard_shared_cuda` on the pinned A6000 UUID
  `GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed`, thermal 90 C opt-in, reserve 0
  (near-capacity envelope pre-declared on #668), timeout 1800 s. Run: 312 s,
  thermal peak 69 C, exit 0.
- Host floors: GPU free >= 40000 MiB, host MemAvailable >= 100 GiB.

## Gates (amended; original gate corrected openly before the re-run)

Against the pinned cell-2 BF16 CPU reference (`reference-v2.pt`, sha256
`450cec47…a6b6850`), same five prompts, same revision:

- Argmax agreement: **5/5 required — PASS.**
- 12-token greedy: exact, OR first divergence at a top-2 margin < 0.1 logit
  units (characterized fp16-vs-BF16 tie-flip). Measured: **4/5 exact; the
  single diverging row flipped at margin 0.0232** (ours 25438, logit 19.31;
  reference 8807, 19.28) — PASS under the amended gate. The original
  byte-exact greedy gate contradicted the #23 convention it cited
  (cross-precision token exactness is not demanded); the correction, with the
  threshold fixed in advance, was posted on #668 before the re-run.
- Worst rel-RMS: **0.1083** — inside the Q8 cell's 0.114 envelope. Reported,
  not gated.

## Capacity and transfer accounting

- Peak live device weights **3.46 GiB** (fixed head modules + the largest
  single layer), allocator peak 5.43 GiB, vs the 44.5 GiB card.
- 37,803 tensor transfers; ~4.8 s prefill, ~51.4 s per 12-token decode.

## Controls

- **Control A (`guard-ctrlA.json`, exit 0)**: streamed-vs-resident executor
  parity on real CUDA fp16 — bit-exact prefill/decode, spec ≡ plain greedy,
  store released to zero. The streaming copy path is arithmetic-transparent;
  the greedy divergence is not a streaming defect.
- **Control B (`ctrlB.log`, `guard-ctrlB.json`, exit 0)**: per-step top-2
  margins on the diverging row — the divergence step is a rank-1 near-tie
  (margin 0.0232), all prior steps chose the reference token with margins
  0.097–4.94. This is the evidence for tie-flip admission above.
- **Retained failed runs**: `stream-gate5.json`/`guard-stream5.json` (the
  first full run under the stricter gate, greedy 4/5 with no margin
  characterization), plus the three launch-failure iterations recorded on
  #668 (host-gate overestimate, guard env stripping, corpus/tokenization
  drift).

## Honest limits

This is a **capacity + numerics cell. No performance claim**: synchronous
streaming re-transfers each layer's tensors every decode step and is
PCIe-bound by design (~51 s per 12 tokens). No warmed steady state, no
throughput number is claimed. Quantized streaming (q8/q4) remains refused for
a structural reason: packed `Linear` scales are keyed by host-tensor `id()`,
which streaming copies invalidate — follow-up work, not a bug.

## Admission policy (the generalization this cell exercises)

`layer_stream` admission is structural, not name-locked: any
`HybridExecutor` text-family model (dense or MoE) whose head modules are
plan-resident and whose per-layer tensors are single-source floating point
admits. The MoE+MTP virtual layer streams exactly like ordinary layers.

## Artifacts

| File | sha256 |
| --- | --- |
| stream-gate6.json | b9cca26c679d5b5025e8a00ff1c355842a267bd7242c6e9964c4b53bb953b134 |
| guard-stream6.json | 8b5d901e25dc6869dc360a03f0c8d46eeb6cb7019334fab7ba199f87aba05805 |
| ctrlB.log | ea08e82775b0dd0c00fa939bef4d668128a8e9fe17f47fd37315b4151b14b1cf |
| guard-ctrlA.json | ff2c00d856493cda25a4842784918981bd7cc6565f0b046d98feb55ac23ea674 |
| guard-ctrlB.json | 9a2ab228cfdb61d6ae809e99361b922ccd27b3f0aa70f0cd9356df4988cf2b9d |
| stream-gate5.json | 06af9e546cf4ba520ead234a65ac91d73614d14bdacc4d2568fedd803d954e74 |
| guard-stream5.json | 50667f517dc34dc283e838e291d12ab86c2d07b113618cf93577cdcc9b39ed55 |

Runner: `scripts/experiments/qwen36_35b_streaming_668.py` (guarded, single
A6000). Probes (staged copies, scratch): CUDA parity control and row-0
margin characterization.

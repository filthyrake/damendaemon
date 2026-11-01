# Qwen3.6-35B-A3B Q8-packed synchronous layer-streaming cell (A6000)

**Result: PASS — all gates.** The qualified Q8 snapshot (35.6 GiB packed)
streamed through synchronous layer streaming with **streamed == resident
exactness**: packed int8 weights and their fp16 scales stream together, the
packed Linear rebinds its id()-keyed scale lookups after every activation,
and the streamed executor matches the resident executor byte-for-byte on
identical weights, kernels, and prompts.

## Snapshot

`Qwen/Qwen3.6-35B-A3B` at HF revision
`995ad96eacd98c81ed38be0c5b274b04031597b0` (same pin as every #668 cell).
Q8_0-packed on load, 35.6 GiB with scales; host-resident, load 124.5 s.

## Configuration

- `DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD=1` + `DAMENDAEMON_HYBRID_MTP=1`
  (`embed_first`), set in-process (the guard's hermetic child env strips
  `DAMENDAEMON_*`), eager torch FP16 activations, no graphs, context 512.
- Guarded by `scripts.experiments.guard_shared_cuda` on the pinned A6000
  UUID `GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed`, thermal 90 C opt-in,
  reserve 0, timeout 1800 s. Run: 325 s, exit 0.

## Gates (pre-declared on #668, comment 6564957154)

**Parity gate (primary)** — streamed Q8 vs resident Q8, identical load and
eager torch kernels, same prompts:

- Greedy chains: **token-exact 5/5** (same dtype, same kernels — exactness
  demanded, not drift-tolerated).
- Spec-decode through the streamed MTP virtual layer: **identical** to the
  resident run, and the emitted stream equals plain greedy (drafts always
  verified).
- Allocator peak: 2.42 GiB streamed vs 35.86 GiB resident.

**Reference gate** — vs the pinned cell-2 BF16 CPU reference
(`reference-v2.pt`, sha256 `450cec47…a6b6850`):

- Argmax: **5/5 required — PASS.**
- 12-token greedy per the amended tie-flip rule: 4/5 exact; the single
  diverging row flipped at a top-2 margin of **0.0058** logit units (a
  rank-1 knife-edge, sharper even than the FP16 cell's 0.0232).
- Worst rel-RMS: **0.0839** — inside the Q8 gate's 0.114 envelope.

## Capacity and transfer accounting

- Peak live device weights **2.29 GiB** (fixed head modules + the largest
  packed layer with its scales) vs the 44.5 GiB card — and vs the 35.6 GiB
  resident footprint. This is the machinery proof: quantized streaming is
  what makes models LARGER than the card streamable.
- 63,459 tensor transfers (weights + scales), ~2× the FP16 cell's 37,803
  (int8 weight + fp16 scale per packed projection).

## Machinery notes

- Packed scales stream with their weights in `LayerWeightStore` (same
  alias/pop discipline as the weights; accounted in
  `copied_bytes`/`resident_bytes`/`peak_live_bytes`).
- The executor rebinds the packed Linear's id()-keyed lookups **wholesale**
  after every activation via the single `_activate_streamed` chokepoint —
  stale device ids can never linger (ids are addresses, reusable after
  release), and the packed decode cache is dropped per rebind. A missed
  rebind fails loudly as an id miss, never silently mis-binds.
- Eager-CUDA packed admission now probes the Triton projection kernels
  (`quantized_kernels`) — the executor's fused selection was never the
  dispatch path for packed projections, so the old blanket refusal
  conflated two surfaces.
- Q4 admits structurally in `layer_stream`; the model-level Q4 policy
  (dense-rotary scope) still applies upstream of it.

## Retained failed launch

The first guarded run failed at executor admission — the pre-streaming
"packed weights require native GPU or Metal kernels" check fired on the
eager-torch CUDA executor (conflation described above). Fixed in-run;
failure retained in the issue record.

## Honest limits

CAPACITY + NUMERICS cell, no performance claim: synchronous streaming
re-transfers each layer's tensors per decode step (PCIe-bound by design).
A near-card-fitting Q8 model was streamed deliberately — to qualify the
quantized streaming surface, and because models LARGER than the card need
exactly this path.

## Artifacts

| File | sha256 |
| --- | --- |
| q8stream-gate.json | 6acedd316b90f5070becdc4cbe20cb1e20839d4513c7bd245680f109e51f9329 |
| guard-q8stream.json | e9213198726eaf682e6d325be22f7f1da96118414daee8bc9cf20736ba86ffc1 |
| q8stream.log | 7efcd9f6dc3e11fe158c19229a974f9a1ef0f11cc43cc8622b1eebe37dd925cf |

Runner: `scripts/experiments/qwen36_35b_q8_streaming_668.py` (guarded,
single A6000, phases: streamed + resident control).

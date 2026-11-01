# Qwen3.6-35B-A3B MTP speculative decode — transcribed head, exact gate PASS (#668 MTP cell, 2026-11-01)

The checkpoint's 19 `mtp.*` tensors are the only piece of the target no
upstream reference pins: eager Transformers **drops them at load**
(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). The head was
transcribed structurally (DeepSeek-V3 MTP as carried by the Qwen3.6
layout: `fc` over the two pre-normed inputs, one full-attention MoE layer
— same 256-expert FFN surface as the main model — final norm, shared
embeddings/head) and qualified by the property that is load-bearing for
correctness: the draft is **always verified against the main head's
greedy argmax**, so the emitted stream is exact regardless of head
quality.

## What shipped (branch `issue-668-mtp`)

- `load_weights` transcribes the MTP block under a **virtual layer index**
  (N = num_hidden_layers): the existing attention/FFN/packed-quantization
  machinery applies with zero duplication. Opt-in
  `DAMENDAEMON_HYBRID_MTP=1` (default loads stay byte-identical);
  `DAMENDAEMON_HYBRID_MTP_DROP=1` skips the head. Refused with CUDA
  graphs, token gating, quantized KV, or hybrid stage ranges.
- The MTP block runs **eager masked attention with the architectural
  position-0 hole** (the block never sees the first token's predecessor —
  position 0 has no MTP key) through its own KV cache.
- `spec_decode`: chunked MTP KV prefill over the prompt, then per
  iteration one main decode step verifies the previous draft; accepted
  drafts emit two tokens per main step. Returns (stream, accepted, steps).
- The **fc concat order was the one genuine semantic unknown** (no
  upstream code pins it); both orders are implemented behind
  `DAMENDAEMON_HYBRID_MTP_CONCAT` and the gate measures which is real.
- Loader: `mtp.*` tensors join the state and the host weight estimate
  only under the opt-in (default loads unchanged).
- 7 CPU tests: opt-in/drop/env surfaces, virtual-layer transcription,
  dedicated-embeddings refusal, and the load-bearing property —
  **spec == plain greedy exactly on identical weights, under both concat
  orders** (plus the concat-order sensitivity check on random weights).

## A6000 gate — ALL PASS (guarded, exit 0, 936 s, peak 73 °C)

Runner `scripts/experiments/qwen36_35b_mtp_gate_668.py`; three sequential
subprocess phases (each loads the 34.76 GiB q8 weights fresh — two
executors cannot coexist): plain reference (MTP dropped), then spec under
each concat order. Same pinned snapshot/quantization as the cell-2 gate
(`995ad96e…`, q8_0, Triton, `--no-graph`, context 32768). Prompts: the
five cell-2 gate prompts (12 tokens) + one 32,256-token C-series corpus
prompt (512 tokens).

| phase | exactness vs plain | acceptance (shorts/long) |
|---|---|---|
| spec `embed_first` | **6/6 token-exact** | **45/45 + 511/511 = 556/556 (100%)** |
| spec `hidden_first` | **6/6 token-exact** | 0/45 + 0/511 (0%) |

- **Exactness passed under BOTH orders** — correctness-by-construction
  verified on hardware: even a zero-acceptance (wrong-order) draft never
  perturbs the emitted stream.
- **The concat order is pinned empirically**: embedding-first, hidden-first
  — the trained head accepts every draft in the correct order and none in
  the reverse. A ~0% wrong-order acceptance rate is exactly the signature
  of fc-input permutation; the asymmetry is the measurement, not a
  coincidence.
- 100% acceptance on this corpus is an upper bound, honestly bounded by
  the corpus itself: the long prompt is the repeated-paragraph C-series
  corpus (highly self-predictable) and the shorts are similarly regular.
  Acceptance on diverse natural text will be lower; measuring that is a
  follow-up, not a claim made here.

## Honest limits

- No performance/speedup claim — shared-card wall-clock; the MTP loop is
  eager single-slot. Acceptance is draft-quality evidence. Throughput
  qualification of the accepted-draft loop (batched, serving-integrated)
  is future work.
- The transcription is structural: with upstream dropping `mtp.*`, the
  only external anchors are the tensor inventory (shape-exact, all 19
  consumed) and this gate's exactness + acceptance asymmetry. A future
  upstream MTP reference would be the cross-check to run against.
- One earlier launch attempt failed before any GPU work (the loader's
  `mtp.*` state filter — fixed; both the failed and clean driver runs are
  described in #668's launch comments).
- MTP adds ~1.4 GiB over the plain executor (draft head + its 640 MiB
  KV at 32K); peak allocator recorded in the phase reports.

## Artifacts (sha256)

| file | sha256 |
|---|---|
| mtp-gate2.json (driver + exactness verdict) | `3baa06291ab215c017d7fd7667ad422959ba80217244d7ad99971faab1cfe72d` |
| mtp-gate2-plain.json | `080631bab072eab2e491de6d32a36dae34e467fb470fe75754bf79bde5511ef4` |
| mtp-gate2-spec_embed_first.json | `b9d6d045bcc5f93fa2bf20a73bed20a20feede6b692198b38098dca37a766cec` |
| mtp-gate2-spec_hidden_first.json | `553b273dcaf8ad497bcc75b082d3b7fc65ca671e09247b1f0ac3cda339a2b05f` |
| guard-mtp2.json | `56fad64214f0cd9756a2ef361b01db30f91b4fb3538b53e5d4e36e3e9c92a940` |

Phase logs alongside. The plain phase's 512-token long digest `cc380b42…`
reproduces the C-series digest exactly — the reference stream is the
same corpus anchor as every prior 32K cell.

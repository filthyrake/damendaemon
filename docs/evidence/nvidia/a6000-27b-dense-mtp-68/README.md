# A6000 27B dense-MTP speculative decode gate — PASS (#68 decode lever, 2026-11-01)

**Result: PASS — 12/12 exact, concat order pinned, dense draft qualified.**
The C-series model's dense MTP draft head now transcribes and runs under the
exact-by-construction self-speculative path.

## The head

The 27B snapshot (Qwen3.8-27B, rev `1d4bf0f2…`) carries 15 `mtp.*` tensors:
fc + pre-norms + one full-attention layer with a **dense** SwiGLU FFN
(`mtp.layers.0.mlp.{gate,up,down}_proj` — no router/experts, unlike the 35B
MoE head from the #668 cell). `mtp_num_hidden_layers: 1`, shared embeddings.

## Changes (branch dense-mtp-27b)

1. `hybrid.load_weights` — the MTP transcription admits dense checkpoints:
   the draft FFN mirrors the main checkpoint (dense gate_up/down under the
   virtual index N; `_mlp_half` dispatches structurally on router absence —
   zero forward-path changes). Opt-in contract unchanged
   (`DAMENDAEMON_HYBRID_MTP=1` / `_DROP=1`), now enforced for dense
   checkpoints carrying `mtp.*` too (previously silently ignored at load).
2. `qwen2.ModelWeights` — the safetensors read filter and byte audit
   admitted `mtp.*` only for `qwen3_5_moe_text`; now both hybrid families
   (`qwen3_5_text` included). Found live: the first spec attempt refused
   via the runner's `mtp_loaded` assert — the head never reached
   `load_weights`.
3. `weight_cache.PreparedCache` — the MTP opt-in is threaded into the cache
   identity (`mtp: true` in the manifest): the opt-in changes the prepared
   tensor set, not the source files, so without it a head-less entry would
   be served to an MTP load (and vice versa) on any cache-eligible dense
   q4/q8 load.

## Measured (guarded A6000, q4_0 compensated, 32K context, kernels triton,
three separately guarded phases + local verdict; guard 0 each)

**EXACTNESS — the pass gate: 12/12.** Every spec stream byte-equals plain
greedy (sha256 of token ids): 5 short prompts + the 32,256-token C-series
corpus at 512 greedy tokens, under BOTH concat orders.

**ACCEPTANCE — the measurement (accepted drafts / main steps):**

| | short prompts (5×12) | long32k (512) |
|---|---|---|
| embed_first | 39/55 (71%) | **511/511 (100%)** |
| hidden_first | 1/55 (2%) | **0/511 (0%)** |

The asymmetry pins `embed_first` as the true fc concat order for the dense
family (the #668 precedent; no upstream code pins it — Transformers drops
`mtp.*` at load). The 100% at 32K is on the pinned corpus's deterministic
repetition; the 71% on general short text is the honest headline. Wall-clock
is deliberately NOT a claim: `spec_decode` is the verifier harness (one token
per main step; 417.5 s vs 406.0 s plain at 32K under the shared guard). The
acceptance rate is draft-quality evidence — serving integration (multi-token
acceptance per step, batch slots) is separately declared future work.

## Tests

`tests/test_hybrid_moe.py`: 6 new dense-MTP cases (opt-in refusal, drop,
transcription shapes + no router on the dense draft, spec ≡ plain greedy
under both orders on CPU, requires-head refusal) — 25 passed + 8 subtests
in the file. `tests/test_weight_cache.py`: MTP-keys-its-own-entry test.
Touched-file batteries: 55 + 1 new passed. The 5 pre-existing
test_dense/test_cpu_hybrid_checkpoint failures on this machine (missing
torchvision / phi3 config) are identical on clean main — not from this
branch.

## Artifacts

| File | sha256 |
|---|---|
| mtp-gate.json (verdict) | c18893f3c75dff85126edcdf324b3a0559129f3edf433a2a833837990935dbe3 |
| mtp-plain.json | 2070a7c229b0488e8e229a95ca2c13f790ed90b6f38735295847a18cc083404c |
| mtp-spec-embed-first.json | 28620aea9dae543f6e291690af33c14f48abb7ca414d258934040864a1bec7a1 |
| mtp-spec-hidden-first.json | 26123375ee68de435d3037da606db44a58ea6a20a51a5942bfc73c7106684069 |
| guard-plain.json | 5e646bfc801d31e001ae1ef74a7d935aa8b5ecf8bd2fb3c3065f5438f2d9ecea |
| guard-spec-embed.json | 21b6bbea32f0a0df96a69e8a45fdb766a1f437811ee60ca6beabecb21c1a516a |
| guard-spec-hidden.json | 8eac7a1c076435c73ec957b7782463b5a3a206fe2f939de3d285912fe1d2f078 |
| runner.py | 00d331c5f5df25e14d15aa143ac0d3ad74abb9f08e35e07505c96e5dc2e9aeef |

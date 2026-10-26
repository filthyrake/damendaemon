# TokenGate gate module in damendaemon (#641) — evidence

Child B of the TokenGate evaluation epic (#639), directly downstream of the
#640 hardware evaluation on this same 3090.

## What was built

- `damendaemon/token_gate.py` — the gate module: token scoring from the
  layer-L phase boundary (LN2 → branch MLP with absorbed readout), per-key
  E2EGate bias (article-axis RMSNorm + `softplus(gate_scale)`), top-K
  selection (aggregate + topk), per-head query temperature
  `r = log(m)/log(p)` (m = current valid count, clamped ≥ 2), the decode
  head-mean fold, and a zero-initialized untrained mode that selects
  nothing. Weight loading from the upstream `gate_step*.pt` format
  (`state_dict` + `metadata`, gate head, bias head, per-bias-layer scales,
  fp32-stored scoring branch).
- `damendaemon/hybrid.py` — gated whole-prompt prefill (`gated_prefill`),
  gated chunked prefill (`gated_prefill_chunk`, span validated against the
  whole prompt, phase 2 deferred to the final chunk), per-slot gate state
  (`gate_valid` hole mask, `gate_nvalid`, `gate_fold`), gated eager decode
  branch with the upstream fold as an additive mask, and
  `_gate_mark_ungated` for plain prefills. All production paths are
  byte-identical with `DAMENDAEMON_TOKEN_GATE` unset.
- `tests/test_token_gate.py` — CPU property suite: chunked gated prefill
  equals whole gated prefill (logits, kept sets, recurrent state, decode
  continuation) across chunk geometries; untrained gate falls back to the
  standard path byte-exactly at several K; decode continuation parity with
  a full-attention reference on the hole-mask/fold path; gate-math unit
  tests (bias normalization, selection, ratio). 14 tests, green.
- `scripts/experiments/token_gate_drift_641.py` — the drift record below.

## Numerical reference record (drift.json / drift.log)

Host: rocky10-desktop (RTX 3090, power-capped at 280 W per the #640
incident follow-up; peak draw during this cell was transient single-GPU
eager prefill only). Reference: upstream `token-gate-llms/TokenGate` @
`6e51aac` loaded through its own `load_gate`; checkpoint
`liana313/tokengate-niah-qwen3.5-2b` `gate_step18000.pt` — the paper's
hybrid Qwen3.5-2B gate (gate at DeltaNet layer 1, bias layers {3,7,11,15,19,23}).
Input: a real NIAH 32k haystack body (2,048 tokens) with a non-prunable
prefix/suffix, embedded by our executor, boundary from our real hybrid
prefill path (eager, layer 1).

Scope is component level, stated honestly: upstream's own hybrid
model-forward is broken as shipped (#640: `layer_type` vs `block_type`
dispatch, KV-under-gate requirement, stale 4D mask), so a model-forward
parity is not producible from upstream main. Every gate component is
compared on the same real activations:

| component | ours | upstream | relative-RMS drift |
|---|---|---|---|
| LN2 boundary normalization | `HybridExecutor.norm` | `Qwen3_5RMSNorm` | **0.0** |
| scoring branch | `TokenGate._branch` | `gate_branch` MLP | **0.0** |
| E2EGate bias (article rows) | `TokenGate.bias` | `E2EGate.forward` | **0.0** |
| top-K selection | `TokenGate.select` | `aggregate`+`topk` | **exact index-set match** (128) |
| kept-ratio temperature | `TokenGate.kept_ratio` | `_ssmax_ratio` | **0.0** |

Mechanism evidence on the real hybrid (no upstream comparison — their
model forward is blocked): gated prefill over the 2,067-token prompt kept
147 tokens (128 article + 19 outside the span), finite logits, 8 finite
decode steps.

### The Qwen3.5 RMSNorm pitfall (recorded for reuse)

The first drift run showed relative-RMS 2.70 on the LN2 component. Root
cause was the reference, not the implementation: Qwen3.5 uses the
Gemma-style offset norm (`Qwen3_5RMSNorm`, multiply by `1 + w`; verified
empirically against `transformers` 5.19 on the live model), while the
reference initially used classic `Qwen3RMSNorm` (multiply by `w`).
damendaemon's `take() + 1` storage with a plain multiply is exactly the
`(1 + w)` semantics — correct. Any future norm-parity work on Qwen3.5
checkpoints must use `Qwen3_5RMSNorm`, eps from the model config
(1e-6 here, matching the executor's `self.eps`).

## Divergence note

#640 recorded four upstream ecosystem bugs; the hybrid inference path is
unusable as shipped, so this port could not be validated model-forward
against upstream. End-to-end gated-accuracy claims on hybrid checkpoints
(hybrid NIAH/RepoBench) remain unqualified until upstream fixes its
inference path; the #639 go/no-go gate inherits that caveat for child C
(#642, serving integration) and D (#643, 32K payoff cell).

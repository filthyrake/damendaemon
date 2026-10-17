# GPT-OSS primitive qualification

Tracking: [#39](https://github.com/filthyrake/damendaemon/issues/39). The initial reports below qualify primitives. Subsequent full-model and raw
HTTP qualification is recorded separately at the end. Text Harmony chat is
now qualified in `harmony/`; function handoffs remain unqualified.

Checkpoint: `openai/gpt-oss-20b` at
`6cee5e81ee83917806bbde320786a8fb61efebee`, HF-format root shards,
13,761,316,904 bytes. Official architecture/format reference:
[openai/gpt-oss](https://github.com/openai/gpt-oss/tree/7b583341fe16729127f6d5b94a7b09ccae97e1a1),
Apache-2.0. Implementation is independent; no upstream runtime dependency or
source code copied. The preparation script preserves the checkpoint license.

A2 evidence:

- MXFP4 decoding matches an independent CPU FP64 value-table/ldexp reference
  exactly for all 16 value codes and all 256 scales, including reserved NaN,
  signed zero, subnormals and overflow, for FP32 and BF16 output.
- Routed projections cover shared and per-route inputs, biased projections,
  repeated experts, non-power-of-two widths/output sizes, live expert changes
  during CUDA graph replay and invalid indices without out-of-bounds reads.
- Real gate/up and down tensors from layers 0, 11 and 23, experts 0 and 31,
  match independent CPU FP64 matrix products at the stated script tolerances.
- Attention sinks pass 36 independent FP64 cases across full/windowed attention,
  split reductions, slot IDs and dominant/negligible sinks. Graph replay follows
  changed sink values. Split reduction includes exactly one global sink term.
- The existing Qwen3-0.6B FP32 checkpoint reference replay passes after changing
  the shared attention kernel. CPU/HTTP regression suite: 218 tests pass.

The initial routed projection takes 3.891 ms versus 1.649 ms for an already
expanded, preselected FP32 baseline. Direct FP32 bit construction reduces the
packed result to 2.702 ms (baseline 1.652 ms in that run). This is one first-layer
gate projection, one input and four selected experts, not generation latency.
A simpler exact scale construction plus paired-nibble reads reduces the
projection further to 2.217 ms (FP32 baseline 1.649 ms), a 43% improvement over
the first packed implementation. The exhaustive format checks still pass.
The packed **32-expert bank** occupies 282,009,600 bytes; the baseline's expanded
**four selected experts** occupy 265,420,800 bytes. The baseline excludes routing,
selection and unpacking costs, but the packed path still needs speed work.

Reproduction scripts: `prepare_gpt_oss.py`, `verify_mxfp4.py`,
`verify_mxfp4_checkpoint.py`, `verify_attention_sinks.py`. GPU scripts explicitly
restrict execution to the authorized A2. Reports preserve precision, cases,
measured samples and numerical errors instead of treating successful import or
synthetic execution as model qualification.

## Full checkpoint and raw service

`gpt-oss-reference-replay-a2.json` records 72 passing layer/logit, graph decode,
window-boundary and slot checks against the independent CPU reference generated
by `scripts/reference_gpt_oss.py`. The reference uses unmodified Transformers
5.18 decoder layers, eager attention and FP32 arithmetic, with independent
FP64 table-based expansion of only one layer at a time.

`gpt-oss-service-a2.json` records raw HTTP streaming, two-request batching,
disconnect recovery, graph reclamation and model swapping/host restoration.
`scripts/replay_gpt_oss.py` and `scripts/verify_gpt_oss_service.py` reproduce these
gates. Exact pins, context/precision limits and initial slow-prefill timings
are described in `docs/architecture-parity.md`. No Harmony or cross-engine
performance parity claim follows from these checks.

Grouped prefill is qualified in `gpt-oss-grouped-buckets-reference-a2.json`
(72 checks), `gpt-oss-prefill-http-a2.json` (four alternating full-HTTP rounds,
16/32/129/224-token prompts), and `gpt-oss-grouped-service-a2.json` (129-token
prompts through streaming, cancellation, batching and hot swapping). Exact FP16
unpacking across all format values and compensated products at safe-range
boundaries are recorded in `mxfp4-fp16-matrix-a2.json`. These supersede the
initial routed-prefill performance, while preserving that baseline for comparison.

The `decode/` reports qualify the subsequent smaller packed reduction tiles:
compiler register counts, exact format and matrix checks, 72 independent model
checks, and alternating full HTTP requests using separate baseline/optimized
CUDA graph sets. Reduction order changes, so numerical reference gates and
exact generated-response checks are both retained.

The `harmony/` reports qualify text chat separately: exact prompt IDs against
Transformers, streaming channels against the official Rust Harmony parser,
actual native conversations, Unicode, token caps, stop strings, response-history
round trips, concurrent requests, cancellation and host/GPU swapping. No runtime
dependency on the reference parser was added. Tools, structured chat and chat
logprobs remain rejected.

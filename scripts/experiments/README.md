# Inference kernel experiments

These are opt-in research programs, not serving backends. They do not participate
in runtime backend selection. Run only against the named, authorized test GPU.

## Compensated Tensor Core MXFP4 route (rejected)

`probe_mxfp4_tensorcore.py MODEL OUTPUT.json` compares the existing packed
projection against `mxfp4_tensorcore.py`, which pads one expert route to a 16-row
Tensor Core tile, expands packed values inside each K tile, and uses the existing
high/low FP16 input decomposition with FP32 accumulation. Checkpoint scales must
lie in the exact FP16 range. The probe validates the range before launching.

On the A2 this layout was numerically close but about 6.7–8.2× slower than the
existing chunked FP32 reduction for tested gate/up/down projections. It was
rejected, not wired into the executor. That result concerns this specific layout;
it does not establish that every Tensor Core approach is unsuitable.

Reproduce from the repository root with `PYTHONPATH=.` and the A2 UUID in
`CUDA_VISIBLE_DEVICES`, using the same pinned GPT-OSS checkpoint documented in
issue #39. The probe tests layers 0/23, one/three input tokens, top-4 routing and
both expert projections, including the input-split launch in graph timing.
Evidence: `docs/evidence/gpt-oss/normal-scales/mxfp4-tensorcore-experiment-a2.json`.

## GPT-OSS projection accuracy controls

`probe_gpt_oss_projection.py MODEL BOUNDARIES.pt --output REPORT.json` compares
native compensated projection, IEEE FP32, separate high/low accumulation,
FP64 tile summation and TF32x3 against independent saved output states. Optional
`--modes expert_ieee attention_ieee --positions 787 788 789 790 1024` separates
expert from attention projection effects. Positions are zero-based. Each case
warms once before its measured run; these are exploratory single measurements,
not a production benchmark or a general numerical qualification.

On the A2 sensitive 789-token prefix, only IEEE FP32 and TF32x3 among the first
five policies recover the independent greedy token; their measured prefill time
is 4.80 and 11.46 seconds versus 2.53 native. Neither is enabled in production.
Attention-only IEEE FP32 recovers all five boundary outputs at much lower cost
(2.75 seconds at the sensitive prefix), while expert-only IEEE does not fix it.
`replay_attention_projection.py` runs the existing full-layer, teacher-forced
and slot-isolation gates with this attention-only substitution. It does not
change the gate thresholds or runtime policy.

Boundary references are generated with `reference_gpt_oss_boundaries.py` from
independent layer traces. See issue #41 and the complete retained results under
`docs/evidence/gpt-oss/context-4k/`.

## Fused FP32 sink attention

`probe_sink_attention.py --bounded --output REPORT.json` compares full and
window-128 attention at 129/513/4096 tokens, including independent selected-row
FP64 checks, alternating warm timings and incremental memory peaks. Omitting
`--bounded` selects the initial layout retained for comparison, which is slower
for the 4K windowed case. `sink_attention.py` explains the sink normalization.

This uses PyTorch's version-specific efficient-attention operator to obtain
log-sum-exp. The bounded layout has now been integrated in `dllm/sink_attention.py`
with a 256-row threshold and model-level/lifecycle verification; the earlier
layouts remain here for comparison. `replay_fused_attention.py` runs the existing numerical
gates using `gpt_oss_fused_attention.py`; optional `--ieee-attention` also applies
the independent attention-projection control. `probe_gpt_oss_projection.py
--fused` checks sensitive output prefixes with the experimental attention path.


## Packed multi-row MXFP4 prefill projection (rejected)

`probe_mxfp4_prefill.py MODEL --output REPORT.json` compares unpack-then-compensated
matmul with `mxfp4_prefill.py`, which expands packed weights inside each Tensor
Core tile. The baseline includes unpack cost. Layer 0/23, gate/up and down,
16/128/512 rows were bit-exact, with selected FP64 product error below 3.22e-7
relative RMS. Six alternating warm timing samples per case showed the fused
layout about 3.9–8.2x slower on A2, despite lower temporary memory. It is rejected
and is not wired into serving. This result concerns this layout and hardware,
not every possible packed GEMM. Evidence:
`docs/evidence/gpt-oss/context-4k/gpt-oss-packed-prefill-projection-a2.json`.

## Resident K-quant projection cache

`kquant_dense_cache.py` keeps optional FP16 projection copies inside each
executor for validation only. Packed weights and single-row decode remain
unchanged. The broader experiment now accepts block-aligned Q4_K/Q6_K widths
under a 4 GiB cap so the Qwen3-1.7B fixture can be evaluated; it is not GPU
admission-accounted and does not change normal runtime qualification.
`verify_gguf_reference.py --experimental-dense-cache` requires a nonempty cache
and checks independently expanded HF logits. Layout and HTTP comparisons have
separate gates. Normal `DLLM_KQUANT_DENSE_CACHE_MIB` remains restricted to its
qualified model scopes in `dllm/kquant_cache.py`.

## Q4 prompt dispatch without resident copies

`../profile_q4_dispatch.py MODEL --output REPORT.json` compares the current
production prompt path with temporary expansion plus dense GEMM on actual model
projections. Expansion is included in every candidate invocation; no dense
weight cache survives the operation. It checks the existing FP16 coefficient
policy and alternates control/candidate/candidate/control timings at 16, 32,
64, 128, 129 and 256 rows. The 129/256-row cases already use the expanded path
in production and provide a timing control. Matrix results do not qualify a
model, a production dispatch change, or a server-level speedup.

The `--tiles` mode instead compares 16/32-row fused tiles against production,
with the same independent arithmetic gate. Real Mistral matrices favor 32-row
tiles at 64 and 129 input rows; other tested sizes do not justify a change.
`q4_small_tiles.py` isolates precisely those cases and projection shapes for
model, layout and HTTP validation. Use `--experimental-q4-small-tiles` in the
reference/layout verifiers or `--q4-small-tiles --native-format q4` in the engine
benchmark. Include `--extra-prompt-tokens 64` in reference and HTTP workloads;
the existing control lengths remain. Normal serving does not import this code.

`../profile_q4_expansion.py` separately measures the long-prompt expansion
operation. `q4_expand.py` loads each packed byte once and writes both decoded
nibbles, retaining the same scale and FP16 rounding. Every projection must be
bit-exact against the independent coefficient decoder before it is timed.
This experiment has no production installation hook and retains no dense cache.

For the normal hardware-bounded 33–64-row policy, use
`--q4-small-tiles --production-q4-small-tiles` in the HTTP benchmark. Its control
server disables only the new tile selector; its candidate is the ordinary
`dllm` entrypoint. This differs from the earlier 64/129-row experimental
override. The normal-path reference/layout verifiers accept
`--require-q4-small-tiles` to reject a run that silently used a fallback.

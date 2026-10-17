# GPT-OSS longer-context memory and numerical investigation

Checkpoint: `openai/gpt-oss-20b`, revision
`6cee5e81ee83917806bbde320786a8fb61efebee`.
Hardware: NVIDIA A2, native MXFP4 weights, FP32 residual/KV policy.

The subsequent [fused-prefill implementation](../fused-prefill/README.md) repairs
the five targeted output prefixes. The original failures below are preserved
as historical controls. This evidence does **not** qualify longer-context numerical parity. Track the
open failure in [issue #41](https://github.com/filthyrake/damendaemon/issues/41).

## Memory change

GPT-OSS prefill partitions independent attention query rows with a 32 MiB
partial/LSE and temporary-output budget. Global positions and slot IDs remain
unchanged. Dead attention and expert intermediates are released before the next
large allocation; expert bias addition reuses the projection result.

For 4096 queries, 64 query heads, 8 KV heads and head width 64, the measured
incremental attention peak falls from **1,157,627,904 to 100,458,496 bytes**.
The latter includes the full 67,108,864-byte output. Both full and 128-token
windowed attention are bit-identical to the unbounded implementation; selected
rows agree with independent FP64 attention to maximum absolute error 2.39e-7.
The full two-slot, 4096-context numerical replay completes within
15,177,883,648 allocated bytes. These are memory measurements, not speed claims.

The HTTP lifecycle test also passes with 4000 prompt tokens, a 4096 context and
two slots: stream equivalence, disconnect recovery, concurrent isolation, graph
reclamation on model swap, host-cache restore and explicit unload/reload. Peak
allocation is 15,191,919,616 bytes. The single measured warm request took 17.60 s
for 4000 input / 16 output tokens; this is not a comparative performance claim.
The existing 72 short reference checks and all 241 CPU tests pass.

Long-prompt Harmony function lifecycle checks also pass at context 4096: the
first request uses 3363 actual prompt tokens and the tool-result turn uses 3414.
The test covers complete JSON handoff, tool-result history, SSE equivalence,
incomplete-call suppression, disconnect recovery, unique concurrent call IDs and
rejection of unsupported forced/strict tool choices. This is protocol/lifecycle
evidence; it does not override the independent numerical failures below.

## Numerical failure retained

The 72-check long replay has 14 failed layer checks (layers 17–23 in each
case), despite passing the tested endpoint logits and teacher-forced tokens.
An independent CUDA FP32 reference with TF32 disabled reproduces the seven
late-layer failures in the shorter case.

Targeted prefix checks expose an actual greedy mismatch at 789 tokens:
native token 1008 versus CPU reference 279, logit relative RMS 0.1465 and
maximum absolute error 2.877. At 790 tokens the greedy token matches but relative
RMS 0.01535 still fails. Bounded and unbounded attention logits are bit-identical
at all five tested prefixes. This is not caused by attention partitioning.

At layer 17, position 788, the reference's fourth/fifth expert scores differ by
4.53e-6. The native accumulated state crosses that routing boundary. Feeding
native preceding-layer state into unmodified reference attention/router picks
the same experts as native. Replacing only normalization, activation, router
projection, or contribution summation order does not resolve the failing output.
Full FP32 prefill projections fix all five boundary outputs in this fixture,
but still fail some internal absolute-error gates. These controls identify a
projection sensitivity; they do not establish a general repair. No numerical
threshold has been weakened and no near-tie routing rule has been introduced.

The ordered-combine control was removed from the runtime after failing. Its
historical diagnostic report is retained. Projection alternatives are isolated
under `scripts/experiments/`, with no production policy change.

## Reproduction

Use the authorized A2 UUID in `CUDA_VISIBLE_DEVICES`, `PYTHONPATH=.`, and the
pinned checkpoint prepared by `scripts/prepare_gpt_oss.py`. Reference generation
requires the independent Transformers environment documented in the GPT-OSS
architecture evidence. Native scripts use the damendaemon CUDA environment.

- `scripts/verify_attention_budget.py --output attention.json`
- `scripts/reference_gpt_oss.py MODEL --prefix-lengths 1025 4088 --output cpu.pt`
- `scripts/reference_gpt_oss.py MODEL --device cuda --prefix-lengths 1025 --output cuda.pt`
- `scripts/replay_gpt_oss.py MODEL cpu.pt --output replay.json`
- `scripts/reference_gpt_oss_boundaries.py MODEL cpu.pt --positions 787 788 789 790 1024 --output boundaries.pt`
- `scripts/replay_gpt_oss_boundaries.py MODEL boundaries.pt --output boundaries.json`

Both failing replay commands deliberately return a nonzero status while saving
the complete report. Large reference tensors are local test artifacts; the
committed JSON reports retain all check outcomes, including failures.

## Further controls (experimental, not serving policy)

Attention-only IEEE FP32 projections recover all five targeted output prefixes;
expert-only IEEE FP32 does not. The attention-only full long replay still fails
five internal absolute-error checks, with maximum error 1.256 and relative RMS
around 1e-5. All other original gates are retained in its report.

The independent CPU and CUDA reference traces themselves differ by up to 1.790
in late hidden states, exceeding the same 0.5 absolute threshold; their saved
endpoint logits differ by at most 7.11e-5 and their greedy tokens agree. This
reference variability is separate from the original native greedy mismatch.
Thresholds remain unchanged while the discrepancy is investigated.

A fused-attention experiment uses ordinary fused attention's log-sum-exp to
recover sink normalization: multiply its output by `sigmoid(logsumexp - sink)`.
Head partitioning bounds full-attention GQA expansion, while query partitioning
trims each sliding-window tile's keys. At 4096 queries, six alternating warm
measurements yield median full-attention 772.17 -> 86.63 ms and window-128
131.30 -> 17.82 ms. Selected independent FP64 rows agree within 8.35e-7 across
all tested sizes. Measured incremental peaks are 92.7 MB full and 77.8 MB
windowed, versus 101.2 MB baseline. The first unbounded windowed layout was
slower and used more memory; its report is also retained. These are component
measurements, not end-to-end performance or numerical qualification.

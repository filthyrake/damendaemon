# Q4 arithmetic isolation

Tracked in issue #23. The expanded CUDA FP16 Qwen2.5-0.5B trace has three greedy
failures in both sequential and packed layer-partition execution, despite RMS
and maximum errors staying within the unchanged numerical limits. These
experiments investigate that gap; they do not mark it resolved or change serving.

The pinned model revision is `7ae557604adf67be50417f59c2c2f167def9a775`, with
Q4_0 coefficients, context 512, four slots and A2/A30 stages split 12/12.
`scripts/experiments/diagnose_q4_head.py` replays the same 282 independent
teacher-forced checks eagerly. Each output head variant sees exactly the same
normalized hidden state within an event. Diagnostic dense copies restore the
stored Q4 coefficients to FP16; the FP32 head then promotes those coefficients
and activations, with TF32 disabled. No variant's output feeds back into the
trace, and no gate or independent target is changed.

| Body projections | Native packed head failures | Dense FP16 head failures | Dense FP32 head failures |
|---|---:|---:|---:|
| Native packed | 3 | 3 | 3 |
| Temporary dense FP16 | 1 | 1 | 1 |
| Native packed + unfused normalization | 3 | 3 | 3 |
| Dense FP16 + unfused normalization | 0 | 0 | 1 |
| Contiguous packed body products | 0 | 0 | 2 |

With native body projections, native and dense FP16 heads fail the same events
61, 135 and 159. The FP32 head fixes events 61/135 but introduces failures at
93/144; event 159 remains wrong. Those new cases include reference FP16 ties,
so removing the final rounding point is not a general solution.

With temporary dense body projections, native and dense FP16 heads fail only
event 135, while the FP32 head fails event 93 instead. This implicates body
arithmetic as a contributor, but does not establish the complete cause. A dense
output head alone does not solve the issue, and a larger dense-body fallback
still does not pass on its own. Neither is promoted. These are correctness diagnostics,
not performance measurements, and they intentionally use extra GPU memory.

Reproduce with the independent tiny-prefix Q4 trace and `--output FILE`, then
repeat with `--dense-body`. `--packed` selects packed prompt admission; the
recorded controls here use the sequential default. All jobs are bounded and
run sequentially without altering the existing services.

The existing experimental `q4_prefill.single_dot` kernel changes body projection
accumulation order from separate low/high nibble products to contiguous K-order
products. It keeps coefficients packed, FP32 accumulation and FP16 outputs,
and retains the native packed output head. That candidate passes all 282 eager
checks without replacing normalization. The explicit `run_q4_body.py replay`
dispatch also passes 282 checks with real CUDA graphs and 22 observed packed
stage calls. This is evidence for a candidate, not proof of complete equivalence
on arbitrary prompts or a production promotion. Its serving and performance
qualification are separate from these isolation controls.

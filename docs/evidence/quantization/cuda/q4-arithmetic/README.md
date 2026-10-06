# Q4 arithmetic isolation

Tracked in issue #23. The legacy CUDA Q4 body arithmetic produces three greedy
failures in both sequential and packed layer-partition execution, despite passing
RMS/maximum-error limits. The investigation below preserves failed experiments
and their measurements. The integrated adjacent-nibble stage implementation
subsequently passes the expanded gate on A2/A30 and A2/3090, including graphs
and packed prompts. It is now the default for the qualified Qwen2.5-0.5B stage
configuration; ordinary single-GPU Q4 dispatch remains unchanged.

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

The explicit candidate passes 19 ordinary same-host HTTP checks with packed
prefill enabled, including streaming, concurrency, cancellation and model swaps.
Changed-weight qualification passes 410 independent checks across four alternating
weight sessions and a fresh recovery after partial-upload failure. Three actual
rebinds preserve captured graph storage; the partial executor is unpublished.
These lifecycle timings are not a matched hot-swap benchmark.

The same contiguous-body candidate passes 282 checks on the single A2 with
graphs as well. However, its matched A2/A30 HTTP measurement rejects it for
production on performance grounds. Both modes keep prompt packing disabled,
capacity four, context 512, 16 input IDs and 32 output tokens; paired/contiguous/
contiguous/paired launches provide ten measured rounds per mode/concurrency.
Text and usage match. Startup/capture and warmups are excluded.

| Concurrency | Body kernel | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Current paired | 28.17 | 162.48 | 196.35 |
| 1 | Contiguous candidate | 33.41 | 859.39 | 37.21 |
| 2 | Current paired | 43.93 | 201.69 | 312.58 |
| 2 | Contiguous candidate | 56.86 | 912.50 | 68.90 |
| 4 | Current paired | 107.63 | 289.04 | 438.81 |
| 4 | Contiguous candidate | 133.22 | 988.18 | 128.48 |

The first candidate's large throughput regression is unacceptable despite its
passing correctness gates. It remains an explicit experiment; production Q4
dispatch and conservative prompt-packing default are unchanged. Reproduce with
`bench_partition_prefill_service.py --compare q4-body --quantization q4_0`.

## Packed-load revision

`single_dot_paired_load` loads each packed byte once and each block scale once,
then reconstructs contiguous K-order values for the dot product. It passes all
282 graph-backed model checks and all 20 synthetic body-matrix checks. The
synthetic checks require exact FP16 output equality with the first contiguous
candidate and retain the independent dense RMS gate.

Warm CUDA-graph profiling on the A2 exposes the remaining cost. For one row:

| Projection (outputs × inputs) | Current paired, µs | First contiguous, µs | Packed-load revision, µs |
|---|---:|---:|---:|
| QKV (1152 × 896) | 8.88 | 79.92 | 19.39 |
| Attention output (896 × 896) | 8.78 | 79.71 | 19.34 |
| Gate/up (9728 × 896) | 59.33 | 629.02 | 145.78 |
| Down (896 × 4864) | 44.81 | 460.32 | 103.18 |

The revision is substantially faster than the first candidate but still
2.2–2.5 times slower than current body kernels on these small-row cases.
Compiler metadata reports 211 registers versus 128 for current kernels, with
zero spills in both. A launch/tile sweep is needed before considering another
serving comparison. These synthetic matrix measurements are not model
throughput claims. `benchmark_q4_kernels.py --body-candidates` reproduces them.

The bounded tile/launch sweep (`--tune-body`) keeps K tiles at 128 and checks
exact contiguous-output equality for every configuration. On the two expensive
one-row projections, its best results are gate/up 131.32 µs versus current
59.27 µs, and down 83.19 µs versus 45.13 µs. Reducing registers or changing
occupancy alone does not remove the regression. No tested setting is promoted.

## Adjacent-nibble layout experiment

A diagnostic byte permutation packs consecutive K positions together instead
of Q4_0's separated half-block nibbles. It preserves integer values, block
scales and packed byte count. The corresponding kernel avoids the per-tile
half-block transpose while preserving contiguous accumulation order.

The initial A2 profile checks two body projection shapes at one/four rows and
three layout-specific launch configurations. Every candidate exactly matches
the original contiguous output. Best one-row timings are gate/up 65.21 µs
versus current 59.13 µs, and down 48.65 µs versus 44.53 µs. At four rows these
are 66.45 versus 60.43 µs and 49.94 versus 44.83 µs. The remaining gap is about
9–11%, a substantial improvement over the earlier candidates but still a
regression. These measurements exclude repacking cost.

The initial measurements above apply only to the matrix profiler and exclude
activation, serving and hot swaps. No checkpoint format, production weight
layout or default has changed.

A subsequent bounded sweep found an exact-output gate/up tile slightly faster
than current kernels (58.05 versus 59.15 µs for one row, 59.43 versus 60.68 µs
for four). The best down tile remains slower (48.52 versus 44.59 µs for one
row). Every timing candidate first passes coefficient-error and exact-output
checks; failed configurations are retained as failures and not timed.

The explicit `adjacent-gate` model experiment changes only gate/up projections;
it still fails three of 282 checks, at events 93, 135 and 159, and is rejected.
`adjacent-all` changes all four body projection types and passes all 282 checks
with graphs and 22 actual packed stage calls. It retains the current output
head. The prototype keeps canonical Q4_0 weights plus an additional packed
cache, so its memory use is higher; it is not a production storage format.
Its explicit rebind hook refreshes stable cached buffers or closes the executor
on failure.

That prototype passes 19 real HTTP checks and 492 independent lifecycle checks:
four alternating weight sessions, fresh recovery after partial canonical upload,
and fresh recovery after a failure on the second packed-cache refresh. Both
partially modified executors are unpublished; captured packed-buffer pointers
remain stable across successful rebinds. Host residency releases completely.
The additional packed cache is **178,913,280 bytes across both stages**. That
extra allocation is specific to this experiment and must not be omitted from
any memory claim. Performance qualification remains separate.

The cached prototype's matched HTTP run uses the same Q4 weights, A2/A30 pair,
context 512, four slots, 16 input IDs and 32 outputs, with prompt packing disabled
in both modes. Paired/adjacent/adjacent/paired launches provide ten measured
rounds per mode/concurrency. Generated text and usage match.

| Concurrency | Body kernel | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Current paired | 28.63 | 165.80 | 192.53 |
| 1 | Adjacent cache | 27.12 | 164.18 | 194.39 |
| 2 | Current paired | 47.66 | 203.05 | 310.38 |
| 2 | Adjacent cache | 52.45 | 206.64 | 303.42 |
| 4 | Current paired | 107.72 | 289.13 | 437.49 |
| 4 | Adjacent cache | 105.17 | 278.98 | 453.68 |

The original large slowdown is gone. These small, mixed differences are treated
cautiously: throughput is about 1.0% higher at concurrency one, 2.2% lower at
two, and 3.7% higher at four. This is not evidence of a universal speedup, and
the extra 171 MiB packed cache remains an important cost of this prototype.

## Existing-buffer prototype

The explicit `adjacent-inplace` variant repacks the 96 existing body-weight
buffers once during activation and again after canonical weight uploads. It
retains zero additional packed-cache bytes. Large prefills expand the adjacent
representation temporarily for the existing dense fallback. Host checkpoint
bytes and their cache format remain canonical Q4_0; device pointers remain
stable for captured graphs. Repacking still needs temporary workspace.

This variant passes 282 graph/packed-prefill checks, 19 live HTTP checks and
492 lifecycle checks, including both partial canonical upload and partial
in-place repacking failures. Failed executors are unpublished, and fresh
recovery matches independent reference outputs. The prototype remains separate
from production dispatch while its complete policy is measured.

The complete existing-buffer policy comparison enables adjacent body products
and packed prompt admission together. The control retains paired body products
and sequential prompt admission. Same weights, A2/A30 pair, context 512,
capacity four, 16 input IDs and 32 output tokens; ABBA launches provide ten
measured rounds per mode/concurrency. Text and usage match.

| Concurrency | Policy | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Existing | 28.34 | 165.71 | 192.55 |
| 1 | Adjacent in-place + packed | 27.62 | 167.09 | 191.01 |
| 2 | Existing | 42.72 | 197.06 | 319.31 |
| 2 | Adjacent in-place + packed | 45.03 | 207.85 | 304.71 |
| 4 | Existing | 104.56 | 282.40 | 447.80 |
| 4 | Adjacent in-place + packed | 63.84 | 239.60 | 527.13 |

At concurrency four, throughput improves 17.7%, first text 39.0% and request
latency 15.2%. Single-request throughput is roughly unchanged (-0.8%); at two
requests it regresses 4.6% and request latency increases 5.5%. This is a combined
arithmetic/admission policy comparison, not a kernel-only speedup. The runtime
integration must independently pass correctness, lifecycle and serving gates.

## Integrated stage implementation

`DLLM_Q4_STAGE_ADJACENT=1` selects the runtime implementation in
`dllm/q4_stage.py` and `dllm/q4_stage_kernels.py`. It is restricted to the qualified
Qwen2.5-0.5B stage geometry, split 12/12, context at most 512 and up to four
slots. Canonical host/checkpoint data stays unchanged; only device body weights
are repacked. The head retains its current path. Rebind refresh now occurs
inside the existing failure boundary before successful metadata publication.
Distributed startup includes the actual layout in its worker-agreement digest.

The integrated implementation independently passes 282 expanded CUDA graph
checks with 22 packed stage calls, 19 real HTTP checks, and 492 changed-weight
and failure-recovery checks on A2/A30. All 96 body allocations and captured graph
storage remain stable across successful rebinds. Failed partial uploads and
partial repacks are unpublished; fresh recovery matches independent references.
These lifecycle times include activation/cache work and are not a matched
hot-swap speed comparison. At this qualification checkpoint the layout and Q4
prompt-packing defaults were still off; promotion follows the network gates below.

The integrated matrix dispatch also passes 88 real GPU cases across A2 and A30:
all four body projection shapes at 1, 2, 4, 16, 17, 32, 64, 128, 129, 256 and
512 rows. Worst relative RMS versus independently restored FP16 coefficients
is 0.000419 (limit 0.001). Every large-prefill fallback at 129/256/512 rows is
bit-exact with the dense FP16 control. QKV cases include bias. These synthetic
checks exercise layout and dispatch boundaries; they are not throughput claims.

The integrated two-rank A2/A30 HTTP service passes 29 checks, including packed
admission, independent output references, cancellation and a 35-second idle
heartbeat interval. Both ranks reject all four deliberately mismatched startup
policies: capacity, context, prompt packing and Q4 device layout (eight checks).

The integrated same-host policy also passes the matched ABBA HTTP measurement,
using the same workload and controls as the existing-buffer prototype above.

| Concurrency | Policy | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Existing | 29.70 | 168.52 | 189.39 |
| 1 | Integrated adjacent + packed | 27.43 | 162.44 | 196.48 |
| 2 | Existing | 55.00 | 204.92 | 307.80 |
| 2 | Integrated adjacent + packed | 53.29 | 196.04 | 321.41 |
| 4 | Existing | 107.93 | 290.38 | 435.35 |
| 4 | Integrated adjacent + packed | 62.87 | 236.95 | 531.14 |

Four-request throughput improves 22.0%, first text 41.7% and request latency
18.4% in this run. The smaller one/two-request differences change direction
between runs, so they do not establish a repeatable low-concurrency speedup.
Both body arithmetic and prompt admission change; all text and usage match.

On the actual A2/desktop RTX 3090 network pair, the integrated layout passes
all 282 expanded independent numerical checks with CUDA graphs and 11 packed
stage calls on each worker. The fixed-trace tensor-transfer harness validates
numerics separately from the resident service, which also passes all 29 HTTP
checks, including packing, streamed usage, cancellation and idle heartbeats.
A deliberately failing Q4 output worker during a four-prompt/64-row packed
prefill fails all four clients and leaves health/new requests returning 503.
There is no successful completion/usage marker for the failed requests.
These network runs use NCCL Socket transport; they are not RDMA qualification.

The matched network ABBA comparison fixes the A2/3090 pair, context 512,
capacity four, 16 input IDs and 32 output tokens. Both body layout and prompt
packing change; all generated text and usage match.

| Concurrency | Policy | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Legacy | 19.23 | 146.17 | 218.27 |
| 1 | Adjacent + packed | 20.83 | 151.31 | 210.80 |
| 2 | Legacy | 27.05 | 171.98 | 365.66 |
| 2 | Adjacent + packed | 39.07 | 181.78 | 345.30 |
| 4 | Legacy | 53.94 | 232.78 | 542.94 |
| 4 | Adjacent + packed | 48.23 | 213.94 | 591.03 |

At four requests throughput improves 8.9%, first text 10.6% and request latency
8.1%. At two, throughput regresses 5.6%, first text increases 44.4% and request
latency increases 5.7%; single-request throughput is 3.4% lower. These costs
are retained rather than summarized as a universal win. An isolated comparison
holding adjacent arithmetic constant is needed to attribute the two-request cost.

The peer-only firewall rules had enforced ten-minute expiry and were removed
early: lab window 2026-10-06 05:25:11–05:31:59 UTC, desktop
05:25:12–05:32:00 UTC. Final listings contain no test rules; the desktop's
pre-existing 10.42.0.0/16 rule remains. Existing services were not restarted.

After these gates, adjacent stage storage and Q4 prompt packing become defaults
within the qualified geometry. The normal same-host HTTP path passes all 19
checks with both policy variables unset, and its status confirms adjacent body
storage. Explicit legacy/sequential controls remain available. The full local
suite passes 444 tests. The network low-concurrency cost remains an optimization
issue; this promotion does not claim performance parity across workloads.

## Isolating the low-concurrency cost

A second network ABBA comparison holds adjacent Q4 arithmetic fixed and changes
only prompt packing. The harness captures the runtime's existing final timing
records, including actual group sizes, without changing production serving.
All text and usage match. Packing-off/on throughput is 215.85/215.75 tokens/s
at concurrency one, 361.05/352.02 at two, and 555.62/596.87 at four. First text
is 18.89/18.66 ms, 31.73/37.53 ms, and 50.94/47.73 ms respectively.

However, **18 of 20 measured requests at concurrency two used single-prompt
prefill even when packing was enabled**. The one packed pair took about 61 ms
from admission to its first token. This exposes a weakness in the original
warmup procedure: one HTTP warmup round per concurrency does not guarantee that
all possible prompt-group shapes execute before the timed rounds. Queue arrival
order may first expose a new shape during measurement. This is not evidence
that every two-prompt packed operation costs 61 ms, nor a clean basis for a new
scheduler heuristic. The raw results remain preserved as observed behavior.

The follow-up benchmark explicitly warms each reachable group size on the GPU
owner before HTTP timing, releases all warmup slots, and records that startup
cost separately. That distinguishes warm steady-state comparisons from possible
first-use shape costs. Earlier reports exclude server startup and HTTP warmup
rounds, but must not be interpreted as guaranteeing every packed shape was warm.

This second peer-only window was removed early: lab 2026-10-06
05:35:34–05:38:57 UTC, desktop 05:35:35–05:38:58 UTC. Test firewall rules were
absent afterward; A2/A30/3090 allocations returned to 0/13403/10835 MiB.

### All reachable prompt shapes warm

The corrected ABBA comparison holds adjacent Q4 arithmetic fixed, warms group
sizes 1/2/3/4 on the GPU owner according to each packing policy, releases those
slots, then runs the same HTTP measurement. Warmup takes 208/803/227/212 ms in
launch order and is excluded from the following steady-state results.

| Concurrency | Prompt packing | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Off | 19.32 | 148.50 | 214.85 |
| 1 | On | 19.11 | 148.78 | 214.60 |
| 2 | Off | 30.03 | 178.53 | 349.78 |
| 2 | On | 23.57 | 174.46 | 358.39 |
| 4 | Off | 50.07 | 231.21 | 548.42 |
| 4 | On | 46.83 | 211.24 | 599.12 |

The two-request penalty does not reproduce in this corrected run. At four,
throughput improves 9.2% and request latency 8.6%. At one, the policies execute
the same path and differ by about 0.1% throughput. At two, six of 20 requests
actually use pairs; at four, 8/24/8 requests use groups of 1/3/2 respectively.
Arrival/group composition also varies between runs, so warming alone is not
claimed as the sole cause of every timing difference. There is no basis here
for disabling two-prompt packing. First-use shape costs remain a separate
performance concern; this benchmark-only warmup does not change service startup.

The third firewall window was removed early: lab 2026-10-06
05:41:29–05:46:04 UTC, desktop 05:41:30–05:46:05 UTC. Final rules and GPU
allocations again match their pre-test state.

## Runtime row-bound experiment

`scripts/experiments/q4_dynamic_rows.py` removes exact-row specialization from
the adjacent Q4 body kernel while retaining tile sizes and arithmetic. The
explicit A2 matrix experiment uses its own disposable Triton compiler cache,
primes both variants at two rows per projection, then measures first dispatch
and warm CUDA-graph execution separately. It does not modify production dispatch.

All 32 cases (four body projections, rows 2/3/4/7/16/17/19/32) exactly match the
qualified static-row FP16 output and pass the independent dense RMS gate. Per
projection, generated assembly variants fall from eight to two. After a tile
shape is primed, new row counts take about 0.2–0.4 ms to dispatch dynamically
versus roughly 280–550 ms for new static-row compilation in the fresh cache.
The first use of a new tile still compiles. These are isolated matrix/compiler
measurements, not whole-model startup or HTTP latency; they do not directly
quantify the earlier disk-cache/serving first-use event.

Warm performance is mostly similar, but the down projection at 17/19 rows
regresses from about 53.9 µs to 66.7 µs (about 24%). Both variants report 168
registers and zero spills there. A blanket replacement is therefore rejected;
larger-row profiling and a narrower dispatch policy are required before model
qualification. No production default has changed for this experiment.

The larger-row sweep adds 36 exact-output cases at rows
32/33/47/63/64/65/95/127/128. The three 896-wide projections retain similar warm
speed throughout. The down projection also retains similar speed at 33–64 rows,
but dynamic bounds regress at 65 rows (157.89→182.35 µs) and 95 rows
(177.25→183.29 µs). These additional negative results are retained.

The explicit `stage-dynamic` candidate therefore uses runtime row bounds for
all qualified 896-wide body projections, and for down projections only at
1–16 and 33–64 rows. The down projection's 32-row and 128-row tile families
keep the static kernel. Larger dense fallbacks and the output head are unchanged.
This policy follows tile-family measurements rather than selecting individual
benchmark rows. It remains experimental.

That candidate passes all 282 independent expanded A2/A30 graph checks with
22 packed stage calls, 19 actual HTTP checks and 492 changed-weight/recovery
checks. The latter includes partial canonical upload and partial in-place
repacking failures. Whole-model performance qualification is separate; passing
these gates alone does not establish a throughput improvement.

The matched A2/A30 whole-model execution benchmark holds adjacent storage,
packed prefill, context 512, capacity four, 16 input IDs and 32 output tokens
fixed. It uses static/selective/selective/static launches, explicitly warms all
prompt-group shapes, and measures 12 samples per mode/batch size. Every generated
token matches. This bypasses HTTP/queueing and is not serving throughput.

| Batch size | Row bounds | Median first token, ms | Median total, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Static | 22.14 | 140.68 | 227.48 |
| 1 | Selective runtime | 23.01 | 139.80 | 228.90 |
| 2 | Static | 23.64 | 136.44 | 469.07 |
| 2 | Selective runtime | 24.53 | 137.89 | 464.14 |
| 4 | Static | 27.55 | 142.77 | 896.53 |
| 4 | Selective runtime | 27.50 | 143.48 | 892.10 |

Warm throughput differences are small and mixed (+0.6%, -1.1%, -0.5%). The
candidate's potential benefit is avoiding specializations, not a demonstrated
warm-throughput improvement. First-shape observations from this shared process
are retained, but later trials reuse compilation/module state from earlier
trials. They cannot establish a fair cold-start advantage. The next harness
uses a fresh process and private Triton cache per trial; host/driver caches are
explicitly not reset.

### Independent compiler-cache model trials

Four separate processes run static/selective/selective/static, each with its own
fresh temporary Triton cache. Host checkpoint/filesystem and driver caches are
not reset. The same model, Q4 coefficients, A2/A30 placement, context and prompt
sequence are used; every first and repeated output token matches across trials.

| Mode | Median GPU activation/compilation, s | Median first sweep across 15 lengths, s | Median repeated sweep, s |
|---|---:|---:|---:|
| Static row bounds | 22.77 | 48.16 | 0.357 |
| Selective runtime bounds | 15.38 | 19.22 | 0.336 |

Activation here excludes Python imports, host weight loading and initial CUDA
context setup, which are not the optimization target. The first sweep uses
lengths 3/5/7/11/13/17/19/23/31/33/47/63/65/95/127. Selective bounds reduce measured
activation/compilation by 32.4% and cumulative first-sweep time by 60.1%. A new
tile family still compiles, and retained down-projection specializations still
incur first-use cost. These results are not end-to-end cold server startup or
HTTP throughput, and they do not imply every prompt is faster by that ratio.

`DLLM_Q4_STAGE_RUNTIME_ROWS=1` now selects an integrated runtime implementation
of that policy. Static and dynamic entry points share one arithmetic body;
32/128-row down-projection tile families retain static bounds. The flag defaults
off during integrated-path qualification. The integrated kernels pass 88 GPU
matrix cases across A2/A30 and the 282-check independent model graph gate. The
full local suite passes 445 tests. The integrated path also passes 19 HTTP
checks and 492 changed-weight/recovery checks, including partial upload and
repacking failures. The shared arithmetic refactor's static control passes all
282 independent graph checks and 22 packed stage calls. Integrated performance
and cross-host qualification remain separate gates before default promotion.

The integrated implementation subsequently passes the same matched whole-model
ABBA benchmark, with identical generated tokens throughout. Adjacent layout,
packed prefill, 16 input IDs, 32 output tokens, context 512, capacity four and
A2/A30 placement remain fixed; only runtime row bounds change.

| Batch size | Row bounds | Median first token, ms | Median total, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | Static | 23.12 | 144.13 | 222.03 |
| 1 | Integrated selective | 22.88 | 140.08 | 228.45 |
| 2 | Static | 24.34 | 138.54 | 461.96 |
| 2 | Integrated selective | 24.27 | 137.34 | 465.98 |
| 4 | Static | 28.68 | 144.37 | 886.63 |
| 4 | Integrated selective | 27.38 | 142.73 | 896.79 |

This run shows no warm-throughput regression, with differences of +2.9%, +0.9%
and +1.1%. The earlier prototype comparison had small mixed differences, so
these results do not establish a general warm-speed gain. Raw measurements are
in `qwen2-q4-runtime-rows-benchmark.json`; this is model execution, not HTTP or
network throughput. The independent-cache first-use results above are still
prototype measurements and must not be relabeled as integrated-path evidence.

### Rotary token-stride experiment

The existing rotary/cache kernel specializes its token count even though that
value is only a query-output address stride. The isolated experimental kernel
uses a runtime stride with unchanged floating-point operations. Production
`dllm/kernels.py` is unchanged.

On A2, 68 cases cover 17 lengths from 1 through 255, with normalization and
precise arithmetic each on/off, head geometry 14/2/64 and context 512. Queries
and entire K/V buffers, including untouched cache regions, are bit-exact.
Within each arithmetic mode, generated assembly variants fall from 17 to one.
ABBA warm-kernel timing gives median dynamic/static ratios 1.000, 0.993, 0.997
and 0.998; the largest individual slowdown is 1.7%. This establishes kernel
equivalence and warm timing only, not whole-model or first-use latency gains.
The raw report is `qwen2-rope-runtime-tokens-matrices.json`; model qualification
is separate before considering any runtime integration.

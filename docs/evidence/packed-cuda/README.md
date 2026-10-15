# Bounded CUDA packed arithmetic gates

Implementation: `976d77f` (projection), `765d40a` (prepared expert and harnesses).
Shared NVIDIA A2, PyTorch 2.6.0 CUDA 12.4. Explicit APIs only; automatic model
routing and placement are unchanged. No throughput or practical-model claim.

- `projection.json`: all 15 format/dtype cases pass, including strided storage,
  row ranges, empty input and immutable packed tensors. Five formats across
  FP32/FP16/BF16. Independent scalar controls for E2M1/E4M3 and CPU packed
  controls for integer/floating-scale formats. Combined tolerance is
  rtol=2e-5, atol=1e-3; largest absolute difference 0.001953125.
- `expert.json`: all 18 format/dtype/weighting cases pass against scalar-decoded
  dense CPU arithmetic, including duplicate/reordered rows, zero coefficients,
  clipping, cast boundaries and empty input. Largest absolute difference
  1.1444091796875e-05. FP32 tolerance rtol=2e-4, atol=1e-5;
  FP16/BF16 rtol=0.02, atol=0.002.
- Each run peaks at 23,068,672 reserved allocator bytes (22 MiB), below its
  128 MiB cap. Corresponding guard reports retain incumbent identity,
  readiness, temperature, free-memory samples and successful completion.
  Runs were sequential, within 60-second guards, with active incumbent slots
  permitted. No service changes or latency-isolation claim.

Local regression: 71 affected packed-format, expert, checkpoint, stage and
model tests passed. These fixtures establish device arithmetic only. Parent
#14 retains routing, placement, official-weight model and practical qualification.

## Routed composition (#225)

Implementation `b780d97`: explicit prepared-bank routing with one range-check
synchronization and one host count vector; IDs, coefficients and row gathers
stay on CUDA. Duplicate routes add, with an optional unweighted shared expert.
One gathered row/result at a time; output and route-sort buffers are separately
admitted. This is host-scheduled device arithmetic, not a fused routing kernel.
`routed.json` passes six FP32/FP16/BF16 cases with/without shared contribution
against independent scalar-decoded dense CPU experts. Includes duplicate IDs,
zero coefficients, empty batches and invalid route rejection. Maximum absolute
difference 7.62939453125e-06; peak reserved22MiB. Guard completed successfully.
Twelve relevant local controls passed. Practical checkpoint/model qualification
and automatic dispatch remain open under #14.

## Explicit placement (#226)

Implementation `ba06823`: preflight the entire CPU-prepared bank and a unique
raw-tensor byte budget before any CUDA copy; retain aliases and compact owned
storage. Thirteen relevant local tests pass. The actual tiny fixture admits
9,411 raw bytes, preserves each packed tensor exactly and passes the six routed
CPU-oracle comparisons after placement. `placement.json` and its successful
guard retain the result. Allocator overhead remains separately bounded.
The first attempt stopped after an incumbent health probe timed out, after four
cases; its partial result and failed guard are retained as `placement-stopped*`.
A fresh guarded retry completed without changing limits or the incumbent.

## GLM feed-forward boundary (#227)

Implementation `b5ff3f2`: explicit CUDA FP32 GLM FFN with sigmoid/correction
bias/additive-epsilon selection and coefficients applied after down projection.
Five dense/sparse/clipped/underflow cases pass independent decoded CPU control;
routing underflow/tie IDs are checked separately. `glm-ffn.json` and its guard
retain actual A2 results. This does not qualify GLM attention, cache or a model.
Thirty-seven affected local arithmetic/GLM regressions passed. The broader
suite completed 1,724 tests at the earlier routed-composition boundary; later
placement/GLM changes have their focused checks, not a claimed new full-suite run.

## GLM stateless latent attention (#228)

Implementation `6ebd961`. `glm-attention.json` passes two packed latent projection
chunk sizes and three selected-KV tile sizes against independently decoded CPU
weights and dense masked attention. Max projection error1.20e-7; attention2.24e-8.
Causal/padding rejection and original-coordinate packed row views across a scale
block boundary pass. Guard completed with22MiB peak reserved memory.26 local
regressions passed. No indexer/cache/model qualification is implied by this gate.

## GLM full/shared cache (#229)

Implementation `91f816a`. `glm-cache.json` passes three chunked full/shared
attention comparisons against decoded CPU weights, with exact selected IDs,
duplicate-slot reorder, cursor checks, injected post-write poisoning and close.
Largest absolute error1.12e-8;22MiB peak reserved.35 focused regressions pass.
This qualifies the tiny prepared attention boundary, not official model quality.

## Prepared CUDA GLM model (#230)

Implementation `668b74d`. `glm-model.json` passes a three-block prepared packed
model's prefill and two incremental decode positions against independently
decoded CPU weights, including duplicate-slot reorder and exact decode argmax.
Maximum absolute logit error9.54e-7. Injected final-block failure poisons the
whole state without publishing a cursor; close releases all owned layers.
Peak reserved22MiB; successful shared guard retained.40 focused local tests
cover model/block/pipeline compatibility. This is a tiny fixture model, not
an official checkpoint, end-to-end serving or practical performance result.

## Whole-model placement (#231)

Implementation `bd65ce9` plus config-copy follow-up. `glm-placement.json` repeats
model logits/reorder/poison/close through whole-model placement rather than
manual copies. The fixture admits3,628 raw bytes and peaks at22MiB reserved.
All CPU component validation and aggregate unique tensor accounting precede the
first CUDA copy. Local controls also verify late-invalid-member rejection,
aliases, preserved budgets and independent model configuration, and copy failure
leaving the CPU model usable.9 focused tests passed before the GPU gate; the
config-copy follow-up passes the three placement tests. GPU result scope is
unchanged; no checkpoint size or performance extrapolation is made.

## Native generation and serialized HTTP (#232)

Implementation `e358cbf` plus timing-label follow-up. `glm-http.json` proves a
local tiny checkpoint loads, is admitted/copied to CUDA and serves ordinary
chat plus SSE through the pinned native text adapter. Both requests release
owned caches, status/health report CUDA scope and close produces503 health.
The fixture intentionally zeroes its output head to control protocol output;
nonzero numerical logits are established by the separate model gate above.
The original API run predates the completion timing-label correction; the
standalone-command gate checks those labels explicitly. Peak reserved22MiB.
38 generation/serving regressions and34 inherited image/pipeline/cancellation
checks pass. No generated-tool quality or official-model claim is made.

## Standalone CUDA command (#233)

Implementation `6349738`, harness correction `2bf7a16`. Run an explicitly
prepared local GLM snapshot with `python -m scripts.serve_cuda_reference`.
The indexed `--device` is required; CPU weight load, device payload, allocator,
cache, context, output and workspace bounds are separate. It binds loopback,
uses one CPU thread by default and retains accepted requests through cleanup
before printing stopped. CPU commands and ordinary production dispatch remain
unchanged. The allocator cap excludes driver/library memory and does not
reserve resources against other workflows; shared checks still require the
external incumbent/resource guard.

`glm-command.json` retains successful actual subprocess startup, chat, SSE with
requested usage metadata, CUDA timing labels and SIGTERM exit0/stopped scope.
The retry guard completed in9.212s, exit0, with4251MiB free before and after.
The child used a128MiB allocator cap and tiny zero-head checkpoint. Initial
harness failure omitted the SSE usage request; that result and guard remain
in `glm-command-initial*`. The correction did not weaken any bounds.
Ten local command/CPU-command/device tests pass. Later generation compatibility
checks cover34 image/pipeline/queue/cancellation tests and38 native text/tool/
HTTP tests; the full model-core suite passed1,736 tests before these later
placement/generation/command additions.

## Flash recurrence and residual-stream primitives (#234)

Implementation `10f3eb2`. `flash-primitives.json` compares actual CUDA FP32
recurrence and hyper-connection reduction against independent CPU scalar
controls. Five cases cover recurrence, chunk continuation, and one/twenty
Sinkhorn iterations; maximum absolute error is1.2e-7. Restoration orientation,
mean head, device admission, budgets, invalid values and input nonmutation also
pass. Peak reserved22MiB under128MiB allocator cap; guard completed in3.876s,
4251MiB free before/after, incumbent healthy.24 local tests pass. These are
primitive checks, not Flash model, context, multimodal or performance proof.

## Flash stateful linear attention (#235)

Implementation `6286b5c`. Actual CUDA kernel-size1/4 layers pass independent
CPU convolution/recurrence/output equations, full/chunked/token continuation,
duplicate/shrink slot reorder, nonaliasing duplicate slots, injected-failure
atomicity, close and admission checks. Maximum absolute output error1.05e-7;
22MiB reserved,128MiB allocator cap. Guard completed3.848s,4251MiB free
before/after, incumbent healthy.19 local tests pass. Full Flash model and
practical workload qualification remain open.

## Flash learned pooled indexer (#236)

Implementation `df98677`. Actual CUDA selected IDs exactly match independent
CPU scalar pool-score controls. Prefix causality, complete-pool/tail layout,
left/right/all padding and earliest-pool ties pass, as do device, hole and budget
rejections. Key error1.2e-7,22MiB reserved,128MiB cap; guard completed3.948s,
4251MiB free before/after, incumbent healthy.18 local tests pass. Validation
syncs and Python token loops remain; no practical-context performance claim.

## Flash transactional sparse attention (#237)

Implementation `a83f0d2`. Dense and packed CUDA NoPE layers match independent
expanded CPU attention for full/chunk/token execution; selected IDs agree
exactly. Maximum absolute output error1.5e-8. Late projection failure preserves
all histories and cursor; duplicate/shrink reorder, retry, close and bounds
pass.19 local tests pass. Guard completed5.076s,22MiB reserved under128MiB
cap,4251MiB free before/after, incumbent healthy. This does not qualify a full
Flash model or practical workload.

## Flash mixed prepared model (#238)

Implementation `5d3ead2`. A three-block linear/sparse/linear CUDA stack uses
packed dense and routed/shared clipped FFNs plus packed sparse projections.
Final logits match independent CPU scalar component equations, max error9e-8;
reordered decode greedy IDs agree, prepared feature input agrees, late head
failure poisons every layer and close releases all state.31 local block/model/
stage/pipeline/image-feature tests pass. Guard completed7.853s,22MiB reserved
under128MiB cap,4251MiB free before/after and incumbent healthy. No official
checkpoint, vision encoder, serving, practical context or performance claim.

## Whole Flash model placement (#239)

Implementation `7e83257`. The same independent mixed-model logit/state gate
passes through whole-model pre-copy admission and placement:5,812 raw payload
bytes, max logit error9e-8,22MiB reserved. Guard completed7.965s,4251MiB free
before/after and incumbent healthy.12 focused local tests verify full-composition
rejection before copying, aliases, detached config, preserved limits and source
usability after allocation failure. Separate allocator/cache/output/workspace
limits remain required; raw bytes are not total memory demand.

## Explicit Flash text/HTTP command (#240)

Implementation `f4db834`. `scripts.serve_cuda_reference` admits local GLM or
Flash snapshots explicitly, preserving separate bounds and serialized loopback
serving. Flash uses nested text configuration and its native text/tool bindings;
CPU batching and image admission are not inherited. Actual tiny on-disk Flash
snapshot startup, chat, usage-bearing SSE and SIGTERM exit0 pass; readiness and
timing identify CUDA Flash correctly. The zero head controls wire output;
nonzero equation proof is the separate mixed-model gate above.19 local tests
pass. Guard completed10.644s with128MiB child allocator cap,4251MiB free
before/after and healthy incumbent. No official model, tool quality, vision,
practical context or performance qualification is claimed.

## DeepSeek projected compression pooling (#242)

Implementation `4afc647`, test-discovery correction `611295c`. Explicit CUDA
FP32 wrapper admits conservative workspace/output storage before payload scans
or allocation; returned state and block starts count toward the output budget.
Bounded reorder preserves independent slots. Actual CUDA ratios1/2/4/8/128
match independent scalar pooling within1.2e-7; chunk continuation is exact,
including overlap/tails, with failure nonmutation and preallocation rejection.
Eight local tests pass. Guard completed3.375s,2MiB reserved under128MiB cap,
4251MiB free before/after and incumbent healthy. Projection, QAT, indexer,
attention/model and performance qualification are separate remaining gates.

## DeepSeek residual mixing and learned head (#243)

Implementation `5fb30b4`. Explicit CUDA FP32 wrappers admit geometry/static
parameters and conservative workspace/output before payload scans. DeepSeek's
projection-then-normalization and learned final-head equations remain distinct
from Flash. Actual CUDA scalar reduction/head error2.4e-7 at1/20 Sinkhorn
iterations; restoration orientation, nonmutation and budget/device rejection
pass.11 local tests pass. Guard completed3.285s,22MiB reserved under128MiB
cap,4251MiB free before/after and incumbent healthy. No model/performance claim.

## DeepSeek compressed-key selection (#244)

Implementation `e83f56a`. Explicit CUDA FP32 selection admits storage before
scans, validates only causally visible keys and rejects nonfinite visible scores
before ranking. Six actual CUDA chunk/scaling cases match independent scalar IDs
exactly and scores within1.2e-7. Lower-ID ties, future NaN masking, visible poison/
overflow rejection, empty caches/queries and bounds pass.15 local selection and
attention regressions pass. Guard completed3.407s,22MiB reserved under128MiB cap,
4251MiB free before/after and incumbent healthy. QAT and integrated attention/model
qualification remain separate.

## DeepSeek complete compressor (#245)

Implementation `9282a5c`. Explicit CUDA composition admits full projection/pool/
normalization/rotation/simulation storage before scans and rejects intermediate
nonfinite values, including finite pooled values whose variance overflows.
Six actual CUDA ratio4/8 cases with no transform, KV FP8 and index FP4 simulations
match independent scalar pool/norm/rotation plus CPU simulation controls within
4.8e-7. Chunk continuation, injected downstream failure nonmutation and bounds
pass.16 local tests pass. Guard completed4.028s,22MiB reserved under128MiB cap,
4251MiB free before/after, incumbent healthy. Integrated attention and official
checkpoint/performance qualification remain open.

## DeepSeek learned packed indexer (#246)

Implementation `157009f`. Explicit CUDA FP32 admission includes packed projection
scratch, query transforms, selection and input-validation storage. Dense and
E4M3/E8M0 projections pass six chunk cases against CPU-decoded projection/scalar
rotation/simulation and scalar ranking: exact IDs, score error below1e-6.
Intermediate overflow and pre-projection budgets fail closed.15 local tests pass.
Guard completed4.091s,22MiB reserved under128MiB cap,4251MiB free before/after,
incumbent healthy. Integrated attention/model and practical performance remain
unqualified.

## DeepSeek selected-KV attention (#247)

Implementation `30d448f`. Explicit CUDA FP32 attention bounds output/workspace
and rejects positive out-of-range IDs before gathering. It validates selected
vectors/scores while preserving negative padding and unselected poisoned cache
slots. Three CUDA chunk cases match independent scalar sink softmax within2.4e-7;
duplicates, large scores, sink-only/all-padding, overflow and input nonmutation
pass.16 local tests pass. Guard completed3.232s,22MiB reserved under128MiB cap,
4251MiB free before/after, incumbent healthy. Full cached attention follows.

## DeepSeek owned cached attention (#248)

Implementation `6bd00ea`, strengthened reorder check `b5bc0f7`. Explicit bounded
CUDA FP32 attention composes dense/E4M3-E8M0 projections with compressed pooling,
learned selection and sink attention. Six ratio0/4/128 cases match CPU controls
with independently expanded selected attention within1.35e-7. Compression-boundary
decode preserves cache storage; duplicate-slot decode, shrink, invalid-index
rejection, late-failure poisoning and idempotent close pass.24 focused local tests
pass. Guard completed8.111s with22MiB peak reserved under128MiB cap;4251MiB free
before/after and cleanup confirmed. This is a tiny numerical/lifecycle gate,
not a full model, practical context, isolation or performance qualification.

## DeepSeek packed routing/shared FFN (#249)

Implementation `10b161c`, corrected fixture `b40d5ef`. Four E4M3/E8M0 and E2M1
biased/hash-routed CUDA cases match independent decoded CPU controls: exact route
IDs and outputs within1.68e-8. Selection-only bias, normalized sqrtsoftplus,
clipping and pre-down route weighting are preserved. Input bounds and invalid
hash IDs reject before gathers.19 focused local tests pass. Guard4.851s,22MiB
reserved under128MiB cap,4251MiB free before/after, cleanup confirmed. Initial
FP4 fixture used invalid sub-32-column groups and was correctly rejected; its
failed report is retained. No model, context, isolation or performance claim.

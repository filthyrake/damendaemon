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

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

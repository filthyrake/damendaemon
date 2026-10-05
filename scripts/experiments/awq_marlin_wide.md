# Wide small-row Marlin experiment

Build `native/awq_marlin.cu` with `-DDLLM_AWQ_EXPERIMENT_WIDE` into a separate
library. The ordinary build stays unchanged. This candidate uses 256 threads,
N=128/K=128 tiles, one block per SM and full shared-memory allowance for at most
eight rows. Larger row counts retain the qualified prompt kernels. It uses the
same packed coefficients, group-128 scale/zero layout, FP32 reduction and C ABI.

Motivation: Qwen3-4B AWQ's normal A2 batch-four profile spends 11.773 ms of
19.773 ms total device time in Marlin. The pinned upstream small-batch choices
also include K=128/N=128/256-thread kernels. That is a tuning hypothesis, not a
performance or correctness claim.

Example isolated build from the repository root:

```sh
nvcc -shared -Xcompiler -fPIC -std=c++17 -O3 --expt-relaxed-constexpr \
  -arch=sm_86 -DDLLM_AWQ_EXPERIMENT_WIDE -Ithird_party/marlin \
  native/awq_marlin.cu -o build-awq-qualified/libdllm-awq-marlin-wide-experiment.so
```

Validate real checkpoint matrices first with `scripts/profile_awq_marlin.py`,
then replay the pinned independent Qwen3 oracle with
`scripts/verify_awq_cached_reference.py --experimental-awq-marlin LIBRARY`.
Keep model-level reference/layout gates and unprofiled serving measurements
separate. Do not promote a candidate based on isolated matrix timing alone.
Track conclusions in GitHub issue #49 and retain failed experiment reports.

## A2 outcome

All 96 real Qwen3 matrix checks pass, worst relative RMS 0.0367%. The complete
independent model oracle passes 128/128 checks, worst RMS 0.6632%. However, the
matching batch-four full-model profile records 12.288 ms Marlin time versus
11.773 ms for the qualified default, and 20.302 versus 19.773 ms summed device
time per step. This candidate is not promoted. No HTTP speedup is claimed from
these profiles; they provide no reason to replace the existing kernel. Reports
are retained in `docs/evidence/awq/qwen3-4b-awq-wide-*.json`.

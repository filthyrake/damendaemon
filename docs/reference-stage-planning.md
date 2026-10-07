# Modern CPU reference storage planning

`reference_stage_plan.inspect_reference_storage(snapshot, family)` audits the
existing GLM, Flash or DeepSeek checkpoint contract and reads bounded tensor headers.
It does not read weight payloads, construct models, hash payload bytes, contact
peers or probe devices. It records ordinary file-change receipts and a metadata
identity; that identity is not proof of model bytes. Header passes do not multiply
with the number of possible partitions.

`plan_reference_stages(layout, ordered_byte_budgets, ...)` selects 2..8 nonempty
contiguous ranges covering every layer. GLM cuts preserve full/shared indexer
groups. Dynamic programming minimizes maximum declared-budget utilization, using
exact rational comparisons and deterministic range-order ties. Each ordered budget
belongs to that stage; budgets must already reserve unrelated process memory.

Accounting includes:

- Actual retained packed weights and promoted parameters, with embedding owned
  by the first stage and the complete head by the last.
- The largest selected physical source tensor as conversion-overlap allowance.
- Every admitted slot's context-dependent KV/index/compressor reservation and
  Flash's fixed recurrent/convolution state.
- Separate declared projection, component workspace, output and transport-buffer
  allowances. DeepSeek's component workspace allowance is not an enforced runtime
  cap; its packed projection workspace remains bounded independently.

These sums are declared tensor storage and allowances, **not measured peak memory**.
Python objects, allocator/OS overhead, derived preparation tensors and unmeasured
arithmetic temporaries are excluded. The report never admits execution or qualifies official weights,
practical context, a remote host, accelerators, network performance or speed.
A no-fit result preserves requested context, slot count and stage limits. Invalid
metadata is an error, not a capacity result. Source receipts are checked before and
after planning, and changed inspection limits trigger fresh header admission.

## Dry-run command

```sh
python -m scripts.plan_reference_host_topology deepseek /local/snapshot \
  --budget-mib 256 384 --context 512 --slots 2 --prefill-chunk 128 \
  --plan-output /local/deepseek-plan.json
```

The same command accepts `glm` or `flash`. Weight, cache, projection workspace,
component workspace, output and frame limits have separate options. The JSON
report states scope and exclusions. Exit code 0 means a declared-storage plan fits;
1 means no fit; 2 means invalid input or an I/O error. The optional output is the
existing host command's ranges-only `{"ranges": [...]}` format. It is written
atomically only for a fitting plan; no-fit leaves an existing file untouched, and
an inspected checkpoint file cannot be replaced. Use the resulting plan with the
appropriate explicit host command and its own identity/resource checks.

Tests compare selected weight accounting and cache reservations against loaded
CPU stages, exhaustive partition choices, exact limits, 32K metadata-only no-fit
results, DeepSeek I32/I64 tables, source replacement, malformed headers, and the
actual command. A 32K metadata calculation does not prove 32K execution.

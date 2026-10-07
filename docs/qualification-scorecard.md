# Qualification evidence scorecards

GitHub #54 owns the qualification contract and #3 the roadmap. JSON scorecards
are supporting evidence snapshots, not a task tracker or benchmark runner.

```sh
python -m scripts.evaluate_qualification_scorecard \
  docs/evidence/qualification/current-software.json --evidence-root .
```

The current software snapshot links pushed implementations. Its unresolved
workload/hardware/checkpoint pins are explicitly `null`, so every row remains
`implemented_unvalidated`. It is not a replacement for the retained historical
hardware reports or a complete measured platform matrix.

Current software evidence includes explicit ragged text/tool cohorts and bounded
serialized HTTP queues for GLM/Flash/DeepSeek, token-boundary cancellation, Flash
still-image HTTP, and stage-local Qwen CPU topology commands with authenticated
three-/four-process fixtures. These capabilities have different execution scopes:
ragged reference cohorts run sequentially, modern reference HTTP serializes model
execution, and the separate Qwen topology owner interleaves bounded active slots.
None establishes a named practical model's quality, GPU support or concurrency SLO.
Remaining implementation and measurement gates live in #51, #54 and #55–#59;
fixture sizes and process counts must never fill unknown qualification identities.

## Version 1 records

The root contains `version: 1` and 1–512 `cells`. Each cell requires:

- `id`, a unique stable cell name; `issue`, a repository GitHub issue URL.
- `availability`: `implemented`, `unsupported` or `blocked`. Implemented cells
  require a full 40-character `implementation_commit`; unsupported/blocked cells
  require `reason` and `availability_evidence` report references.
- `identity`, containing every key below. Unknown values must be `null`.
- `comparators`, a list (possibly empty) of required baseline records.
- Optional `numerical`, `lifecycle` and `measurement` records; absence means
  unvalidated. Optional `reason` preserves the scope or limitation.

Identity keys are `model_repository`, `model_revision` (40 hex characters),
`checkpoint_sha256`, `tokenizer_sha256`, `precision`, `hardware`, `topology`,
`context`, `prompt_tokens`, `output_tokens`, `concurrency`, `prompts_sha256` and
`eos_policy`. SHA-256 fields require 64 hex characters. Workload dimensions are
positive integers; prompt plus output must fit context. Full identity equality
is required for qualification, and matching nulls never qualify.

Gate records contain `result` (`pass`/`fail`), `scope`
(`actual_hardware`/`cpu_fixture`), `method`, `identity` and `evidence`.
The accepted numerical method is `independent_reference`; lifecycle uses
`endpoint_lifecycle`. CPU fixtures and other methods remain explicit
unvalidated evidence for hardware qualification.

Measurement records contain `identity`, pinned `engine_revision`, `scope`,
boolean `workload_passed` and `shapes_warmed`, `decode_timing`, `warmup_rounds`,
`measured_rounds`, `metrics` and `evidence`. Candidate counts have A1/A2 keys;
baselines have B1/B2 keys. Each phase needs at least three warmup and ten measured
rounds. `decode_timing` must be `engine_token_steps`; SSE-event or request-average
proxies do not satisfy this metric. Required metrics are:

- `decode_tokens_per_second`, `aggregate_tokens_per_second`
- `first_output_p50_seconds`, `first_output_p95_seconds`
- `request_p50_seconds`, `request_p95_seconds`
- `peak_allocated_bytes`, `peak_reserved_bytes`
- `cold_start_seconds`, `shape_compile_seconds`, `host_restore_seconds`,
  `disk_reload_seconds`

Missing/null metrics remain missing. Negative, nonfinite, boolean or inconsistent
ordered metrics are rejected. Throughput and measured request/first-output
latencies must be positive.

Comparator records contain `engine` and `availability` (`measured`,
`unsupported`, `blocked`). Measured entries contain `measurement` and an explicit
`exact_text_match` boolean/null. Unsupported/blocked entries require `reason`
and `evidence`; they cannot also contain measured results. Missing or failed
exact-text comparisons are retained and prevent competitive qualification.

Report references are lists of 1–32 `{path, sha256}` objects. The CLI verifies
all report bytes beneath `--evidence-root`, including blocked/unsupported
references; it rejects escaping paths, missing/nonregular files and digest
mismatches. Limits are 16 MiB input, 256 report files, 16 MiB per report and
64 MiB aggregate report reads. It does not fetch URLs or execute report content.
Implementation links point to the supplied Git commit; the evaluator does not
contact GitHub to independently prove commit or issue existence.

## Derived states and verdicts

States distinguish `unsupported`, `blocked`, `implemented_unvalidated`,
`numerical_pass`, `lifecycle_pass`, `performance_measured` and `competitive_pass`.
Every result also preserves individual gate results, unresolved reasons,
comparison records and the input evidence. A performance measurement does not
imply numerical or lifecycle success.

A competitive verdict requires passed numerical/lifecycle gates, complete
comparable hardware measurements, exact cross-engine text, and at least one
applicable measured comparator. Blocked/incomplete comparators prevent a verdict;
unsupported comparators never count as a win. Each cell independently needs
both throughput metrics >=95% of the fastest applicable corresponding baseline
and both first-output percentiles <=110% of the fastest corresponding baseline.
No averaging across failing cells is performed. Overall success requires every
cell to pass; unresolved/unsupported cells cannot disappear into a percentage.

These are validated **attestations**, not proof of experiment authenticity.
Hashes bind report bytes; they cannot prove those reports are truthful. The
synthetic unit fixtures exercise decisions and make no hardware measurements.

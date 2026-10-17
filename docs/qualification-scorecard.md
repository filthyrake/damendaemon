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

Current software evidence includes stage-local packed GLM/Flash/DeepSeek CPU
and explicit prepared CUDA paths, native text/tool multi-process commands and
serialized HTTP queues;
Flash image features are owned by the coordinator and carried through language
stages. The Qwen topology owner separately interleaves bounded active slots.
Ragged native reference cohorts remain sequential, and native HTTP queues are
not continuous batching. Header-only placement accounts for declared storage,
not measured peak memory. Weighted scheduling and durable quota accounting
provide software controls; they do not establish GPU capacity or a user SLO.

The snapshot contains fifteen explicit software cells: nine retained CPU/serving
foundations and six explicit CUDA/model/host/storage scopes. Their pushed commit pins
identify implementations, not experiment revisions. The source/test map below
makes those scopes reviewable; local tests are CPU/loopback controls. Separately retained tiny shared-A2
reports establish only their bounded numerical/ownership/protocol gates, not
practical model or benchmark qualification. Unknown practical model/workload/device values stay
null. No numerical, lifecycle or performance qualification record is synthesized
from fixture dimensions or a test count.

| Software scope | Supporting documentation | Focused controls |
|---|---|---|
| GLM packed stages and native host HTTP | [GLM stages](glm-layer-stages.md), [host HTTP](glm-host-http.md) | `tests/test_glm_host_http_command.py`, `tests/test_queued_glm_host_http_command.py` |
| Flash packed stages and native text/tools | [Flash stages](flash-layer-stages.md) | `tests/test_glm_flash_host_http_command.py`, `tests/test_queued_glm_flash_host_http_command.py` |
| Flash image host execution and HTTP | [Prepared image embeddings](flash-host-embeddings.md) | `tests/test_glm_flash_image_host_http.py`, `tests/test_glm_flash_image_host_http_command.py` |
| DeepSeek packed stages and native DSML HTTP | [DeepSeek stages](deepseek-layer-stages.md) | `tests/test_deepseek_host_serving.py`, `tests/test_deepseek_native_host_command.py` |
| Qwen quantized CPU host topology | [Topology commands](cpu-host-topology-command.md) | `tests/test_cpu_host_topology_command.py` |
| Metadata-only modern storage planning | [Storage planning](reference-stage-planning.md) | `tests/test_reference_stage_plan.py` |
| Weighted prompt/decode service | [Serving](serving.md) | `tests/test_weighted_prefill_turns.py`, `tests/test_client_decode_turns.py` |
| Same-host rates and outstanding leases | [Shared reservations](shared-client-reservations.md) | `tests/test_shared_client_rates.py`, `tests/test_shared_reservation_http.py` |
| Remote owner quota authority and HTTP | [Quota commands](quota-commands.md), [admission](remote-quota-admission.md) | `tests/test_remote_quota_admission.py`, `tests/test_quota_commands.py` |
| Device-resident routing and batched expert composition | [Packed CUDA evidence](evidence/packed-cuda/README.md), [device routing](evidence/packed-cuda/cuda-device-routing.json) | `tests/test_cuda_expert_routing.py`, `tests/test_cuda_packed_moe.py` |

Explicit CUDA scopes retain unknown practical identity fields and do not attach
synthetic qualification records to the software cells. The shared-A2 evidence
includes failed/interrupted attempts, guard limits and cleanup; it does not imply
compute isolation or zero impact on incumbent latency.

| CUDA software scope | Pushed implementation | Focused controls and retained evidence |
|---|---|---|
| GLM/Flash/DeepSeek native host text commands | `34374cb` | `tests/test_cuda_host_command.py`, `tests/test_cuda_host_pipeline_build.py`, [component and command reports](evidence/packed-cuda/README.md) |
| Flash local CUDA vision and CPU-vision/CUDA-language hosts | `6766238` | `tests/test_cuda_host_image_preparation.py`, `tests/test_glm_flash_image_host_http.py`, [image reports](evidence/packed-cuda/README.md) |
| Header-only CUDA declared-storage audit | `e453015` | `tests/test_cuda_host_storage.py`, [explicit limits](architecture-parity.md) |

The remaining measured matrix stays under #54 and its parent model/platform
issues. Existing bounded historical reports remain separate and retain failures.
Production-scale arithmetic, checkpoint quality, physical multi-GPU/multi-host
transfer, practical contexts and concurrency/swap/soak SLOs still need
their own pinned evidence. Any remaining command lifecycle bug is tracked in
GitHub; a software snapshot is not a claim that no bugs remain.

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

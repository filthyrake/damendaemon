# Multi-GPU communication qualification

2026-10-05. Issue #51. These are communication measurements, not inference
throughput, model capacity, or RDMA qualification.

## Same-host A2 / A30

`a2-a30-transfer-probe.json`: exact tensor copies passed in both directions.
CUDA peer access was unavailable. Four-MiB transfers measured median 5.32 GB/s
A2 to A30 and 5.05 GB/s in reverse. The existing A30 service was preserved.

## Cross-host A2 / RTX 3090

Both workers used Torch 2.6.0+cu124, NCCL 2.21.5, and forced Socket transport.
The desktop's wired link negotiated 2.5 Gb/s; neither worker exposed an RDMA
adapter. The final rank reports contain every timing sample and exact tensor
checks for all-reduce and point-to-point round trips, 4 KiB through 4 MiB.
All eight cases passed on each worker. Rank zero measured median round trips
of 0.476 ms (4 KiB), 1.029 ms (64 KiB), 7.824 ms (1 MiB), and 29.300 ms
(4 MiB). Rank one's participation interval is not the round-trip latency.

Initial direct `send`/`recv` calls interleaved with collective barriers caused
native segmentation faults after the first successful 4-KiB all-reduce. The
Python fatal trace pointed at a subsequent barrier. Grouped asynchronous P2P
operations (`batch_isend_irecv`, waiting for completion) passed. This isolates
a working API path; it does not establish the underlying library defect.
The v3 reports retain the first successful run; its final allocations included
Python P2P descriptors retaining the last tensors. Explicitly releasing those
references reduced final reported live Torch allocation to 512 bytes on each
worker in the final run. This is not a zero-allocation claim.

Tests used tiny transient buffers and preserved the desktop's existing GPU
workload. User-approved peer-only runtime firewall rules had a one-hour timeout;
all three were explicitly removed after the probe. The lab's rich-rule list
was empty afterward and the desktop's pre-existing 10.42.0.0/16 rule remained.
No persistent firewall or router configuration changed.

General distributed recovery, cross-host serving, RDMA, and GPUDirect RDMA
remain unqualified. The model partition experiments below are separate evidence;
a successful communication-only Socket test establishes none of these.

## Same-host model partition proof

`qwen2-layer-stages.json` runs Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, FP16, context 512, four slots.
The independent Transformers trace includes mixed prompt lengths, reordered
slots, replacement prompts, batched decode, and a near-context boundary.
All 82 logits checks passed the existing 0.01 relative RMS / 0.5 maximum
absolute / exact greedy-token gates. Worst relative RMS was 0.005811 and maximum
absolute error 0.085938. The A2 owns layers 0–11 plus embeddings; the A30 owns
layers 12–23 plus final normalization and output head. Each stage has only its
12 KV layers (12,582,912 bytes). GPU weight residency is 630,166,528 bytes and
630,168,320 bytes respectively. The host loader still loads the full checkpoint.

`qwen2-stage-single-gpu-regression.json` replays the same trace through the
ordinary A2 graph-enabled executor: all 82 checks pass, worst relative RMS
0.005656 and maximum absolute error 0.070313. The local regression suite passes
367 tests. The shared runtime change extracts hidden-state layer execution from
embedding/head execution; ordinary entry points retain their behavior.

The initial stage implementation lived under `scripts/experiments`; the narrow
serving integration qualified below now uses `dllm/layer_stage.py`. It executes stages sequentially and copies their activations. It
does not yet have a network control protocol, worker recovery, pipeline
scheduling, or a capacity/speedup qualification. The small checkpoint fits on a
single GPU; this proof must not be represented as a larger-model capacity win.

## Separate workers and cross-network model execution

`verify_distributed_stages.py` runs the same contiguous stages in two processes,
using NCCL typed activation tensors and grouped asynchronous P2P with completion
waits. Workers load a trusted local reference trace; no network object/pickle
collective is used. A fixed 32-byte trace/configuration digest is compared before
activation transfers. This is a test harness, not a general request protocol or
model-weight authentication scheme.

The same-host A2/A30 two-process run passes all 82 independent output-stage
checks. Rank zero has no output head and therefore records no local logits
checks; its pass flag reflects the collective result. An intentional trace-byte
mismatch is rejected on both workers before any model activation transfer, with
`ValueError('worker trace/configuration mismatch')`. Those expected-failure
reports are retained as `qwen2-distributed-mismatch-rank*.json`.

The cross-host A2 / desktop RTX 3090 run then passes all 82 output-stage checks:
worst relative RMS 0.006666, maximum absolute error 0.065430, and every greedy
token matches. See `qwen2-crosshost-stage-rank*.json`. Both workers used Torch
2.6.0+cu124 and NCCL 2.21.5 with Socket requested. This establishes actual
cross-network decoder execution with independent correctness evidence, not
RDMA, serving readiness, recovery, capacity benefit, or speedup.

The three approved peer-only temporary rules were reused within the original
one-hour authorization window, with expiry shortened to stay inside that window.
All were removed again immediately after the model proof; both rich-rule lists
were checked afterward. No permanent firewall/router changes were made.

## Q8 stage ownership

The experimental stage filter also retains and renames only the Q8 scales
belonging to its selected layer weights or final head. It leaves the source
model unchanged and rejects unsupported quantization/packed formats. Two host
ownership tests cover remapping, byte accounting, source preservation, and
invalid bounds; they make no GPU performance claims.

`qwen2-distributed-q8-rank*.json` qualifies the same A2/A30 two-process layer
split against independently generated Q8-dequantized reference logits: all 82
output-stage checks pass, worst relative RMS 0.005336, maximum absolute error
0.052735, every greedy token exact. Checkpoint, context, and slots match the
FP16 experiment, but precision differs; these are correctness results, not a
matched-precision speed comparison. Cross-host Q8 has not been qualified.

## Worker failure and fresh restart

A deliberate exception on rank one after its third completed activation transfer
causes the two-worker job to exit unsuccessfully; the launcher terminates its
peer. The initial probe revealed that a terminated peer could leave no report.
The harness now writes an initial `passed: false, final_report: false` record,
and only marks the report final when its own completion/error handler runs.
The repeated failure probe retains both the incomplete peer record and the
injected worker's explicit error. A fresh pair then passes all 82 FP16 reference
checks (`qwen2-worker-restart-rank*.json`). This demonstrates failed-job reporting
and a clean new job; it is not transparent worker recovery or resumable requests.

## Per-stage decode graphs and matched performance

Opt-in `--stage-graphs` captures decode for each supported batch size while
keeping activation transfers outside the graph. Both FP16 and Q8 two-process
runs pass all 82 reference checks. The first single-process benchmark exposed
an implicit Torch capture stream attached to the wrong GPU. Capture now uses
an explicit stream from the stage device's serialized warmup-stream cache;
this also avoids creating extra cuBLAS workspace streams per stage instance.
The final one-process, two-GPU FP16 and Q8 reference runs each pass all 82 checks.

`qwen2-stage-graph-q8-abba.json` measures graph off/on/on/off, six measured
requests per group after two warmups, identical checkpoint, Q8 policy, context
512, prompt 255 tokens, output 32 tokens, single request, and the same A2/A30
12/12 placement. Every generated token matches between modes. Pooled medians:

| Metric | Graph off | Graph on |
| --- | ---: | ---: |
| First token | 31.16 ms | 31.42 ms |
| Full 32-token generation | 768.76 ms | 134.06 ms |
| Decode rate (31 subsequent tokens) | 42.31 tokens/s | 301.15 tokens/s |

This is 7.12x decode rate and 82.56% lower full-generation latency for this
prototype workload. It is not an upstream-runtime comparison, concurrent
serving throughput, or a cross-network benchmark. Timings include host token
selection and stage activation copies; model load/graph capture are excluded.
Prefill is unchanged and does not materially improve in this comparison.

## Same-host HTTP serving integration

Explicit `--layer-split 12` / `DLLM_EXPERIMENTAL_LAYER_SPLIT=12` admits exactly
two visible GPU UUIDs. Serving is limited to qualified Qwen2.5-0.5B geometry,
FP16/Q8, context <=512 and batch <=4. It rejects paged KV, speculation, chunked
prefill, prefix snapshots and unqualified layouts. The existing request scheduler
owns both devices, with output logits and sampling on the final GPU.

The production factory passes all 82 independent Q8 logits checks. Actual CLI
HTTP runs pass 19 Q8 checks, 20 Q8 injected-failure checks, and 20 allocation-
diagnostic checks each in FP16 and Q8. Checks cover independent reference
continuations, four concurrent requests with differing completion lengths, SSE
content/usage, disconnect and slot reuse, malformed-input recovery, native
log probabilities on the output GPU, seeded sampling with penalties, alias
switches, host restore, full release and disk reload. The local suite passes
373 tests.

A validation-only injected exception after the first stage decodes poisons the
whole executor. HTTP returns an error; status records `executor_failed`. The
next request evicts it, rebuilds both stages from the host cache, and matches
the independent continuation. Failed in-flight requests are not replayed. This
is recovery from an ordinary stage exception, not a guarantee of recovery
from fatal CUDA device errors or a disconnected remote worker.

Status now reports weights/scales/KV per stage and allocated/reserved memory
per GPU. Q8 loaded allocations in the ordinary run were 484,069,376 bytes on
A2 and 359,302,144 on A30. Repeated full releases return host model bytes to
zero and remove every stage. Allocation histories independently identify all
remaining allocations as cuBLAS workspaces: Q8 8,519,680 bytes per device; FP16
17,039,360 per device. Repeated release does not grow either footprint. These
are retained library allocations, so total live CUDA allocation is not zero.

## Compatible checkpoint hot swaps

Compatible layer partitions now restore host-cached weights into the existing
CUDA buffers and retain captured decode graphs. Configuration, precision,
weight/scale layouts and alias relationships must match. Other swaps rebuild
normally. `DLLM_PARTITION_REBIND=0` disables this default fast path.

Publication is transactional: the old alias leaves GPU residency before the
first upload, and the new alias appears only after both devices finish. An
interrupted upload closes the executor; the next activation builds fresh stages.
Request lengths reset on every successful switch.

FP16 and Q8 each pass 410 independent reference-logit checks across A/B/A/B
switches and recovery after an injected second-upload failure. The disposable B
checkpoint changes real weights without changing geometry; its provenance is in
[dllm-test-fixture.json](dllm-test-fixture.json). Weight, scale, KV and graph
buffer addresses remain stable across successful swaps. Full HTTP lifecycle
qualification passes 20 checks per precision with the fast path enabled,
including allocation-history checks after repeated release. Retained cuBLAS
workspace footprints remain as documented above.

The matched Q8 HTTP off/on/on/off benchmark uses the same A2/A30 placement,
checkpoint pair, context 512 and batch capacity four. Each group has four
warmup switches and six measured switches; every switch is followed by an
independent expected-output check outside the timed load operation.

| Median host restore | Rebuild graphs | Rebind buffers |
| --- | ---: | ---: |
| Internal activation | 1.507 s | 0.206 s |
| HTTP model load | 1.512 s | 0.211 s |

This is about 7.2x faster HTTP host restoration across 12 measured switches per
mode. It is not a cold disk-load, FP16 performance, arbitrary-model, or
cross-network result. See [matched measurements](qwen2-partition-swap-q8-abba.json)
and the `qwen2-partition-rebind-*` / `qwen2-partition-http-*-rebind-default`
reports in this directory. The local regression suite passes 376 tests.

## Coordinator-driven worker control

`verify_stage_protocol.py` replaces the requirement that both workers read a
reference trace. Rank zero drives requests; rank one reads only fixed 19-word
int64 control records and typed activation tensors. Before requests, both ranks
compare a SHA-256 over normalized configuration, context, precision and the
actual prepared weight/scale bytes. This is identity checking within a trusted
NCCL job, not network authentication.

The bounded protocol admits at most four slots and context 512. It checks
protocol version, session epoch, command sequence, operation, shape, slot
generation and expected cache position before receiving activations. Replacing
a slot advances its generation; release invalidates its live length. Both
workers acknowledge completed GPU work before committing cache ownership.
A rejected command or execution/transport failure poisons the session; a fresh
worker pair is required. NCCL operations have a 30-second process-group timeout,
and qualification jobs have an external 180-second deadline.

A2/A30 FP16 and Q8 each pass all 82 independent logits/token checks across 35
commands, including reordered and shrinking batches and released-slot reuse.
Only rank zero opens the reference trace. Injecting a stale third command
causes both workers to report failure after exactly two completed commands,
before that command's activation transfer. The intentionally failing reports
remain `passed: false`; they are negative-test evidence, not successful model
runs. A fresh worker pair afterward passes all 82 Q8 checks over 35 commands
(`qwen2-control-restart-rank*.json`). Reports for the initial and negative tests
are `qwen2-control-{fp16,q8,stale}-rank*.json`. The local suite passes 380 tests.

This is a coordinator-driven correctness harness, not a production HTTP worker
service. Full logits return to the coordinator solely for validation, adding
communication that a serving design must avoid. No throughput or cross-host
qualification is inferred from these runs. Existing same-host HTTP serving
still uses its in-process partition executor.

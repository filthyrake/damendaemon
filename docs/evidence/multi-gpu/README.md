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

## Reusable transport and token completion

`StageChannel` now owns fixed-size control exchange and completion fences.
Collective completion reads its result on the host before committing ownership;
merely enqueueing a CUDA collective is insufficient for that host-side contract.
For prefill/decode, a typed int64 token reply can instead act as completion:
rank one samples on the output GPU, rank zero receives token IDs, and both wait
for CUDA transfer completion before committing. Release/stop retain an explicit
collective acknowledgement. The channel requires a trusted two-rank NCCL group
whose model identity and timeout have already been established.

The extracted collective path passes all 82 Q8 reference checks. Token completion
passes all 82 checks each at FP16 and Q8, including reordered/shrinking batches,
slot replacement, and exact agreement between returned tokens and final logits.
The extracted token-completion path also rejects the injected stale third
command on both workers after exactly two completions
(`qwen2-token-ack-stale-rank*.json`). The validation harness still returns full logits separately for its independent
numerical comparisons; the performance harness returns only token IDs.

Two separate A2/A30 Q8 ABBA experiments use the same pinned checkpoint, 12/12
placement, context 512, batch one, 255 prompt tokens and 32 generated tokens.
Each group has two warmups and six measured requests, giving 12 measurements
per mode; every warmup and measured output matches exactly between modes.

The initial control-overhead experiment (`qwen2-channel-q8-abba.json`) compares
a fixed-workload worker pair against dynamic validated commands:

| Median metric | Fixed workload | Validated control |
| --- | ---: | ---: |
| First token | 18.62 ms | 20.09 ms |
| Full generation | 104.78 ms | 124.37 ms |
| Decode | 360.46 tokens/s | 297.62 tokens/s |

The subsequent matched completion experiment (`qwen2-token-ack-q8-abba.json`)
keeps validation enabled in both modes:

| Median metric | Separate completion collective | Token reply completion |
| --- | ---: | ---: |
| First token | 18.81 ms | 18.42 ms |
| Full generation | 122.14 ms | 116.49 ms |
| Decode | 299.56 tokens/s | 315.99 tokens/s |

Token completion improves decode by 5.5% and reduces full-generation latency
by 4.6% in the second experiment. Control overhead remains a performance target.
These are sequential same-host worker measurements, not HTTP throughput,
cross-network performance, an upstream comparison, or an RDMA claim. Local
completion-ownership and failed-fence regressions bring the suite to 384 tests.

## Command-driven A2/3090 network qualification

The bounded command channel now passes all 82 independent logits/token checks
at both FP16 and Q8 across the A2 and desktop RTX 3090. Each pair completes 35
commands; only the A2 coordinator opens the reference trace. The desktop is
explicitly passed a nonexistent trace path. Prepared-weight identities agree
between hosts. Stale third-command injection is rejected by both workers after
exactly two completed commands, before the next activation transfer.

The first attempt exposed an incidental NumPy dependency in the experiment's
weight fingerprint. `stage_identity.py` now hashes explicitly bounded CPU tensor
storage directly, without NumPy or a full byte-buffer copy. Its versioned hash
includes configuration, context, precision/format, dtype, shape, weights and
scales. This identity format differs from the earlier experimental fingerprint;
hash values from those formats must not be compared directly. The local suite
passes 387 tests, including identity changes and non-CPU pointer rejection.

The network ABBA benchmark uses the same Q8 checkpoint pair, context 512,
255-token prompt, 32 output tokens and batch one. Both modes retain bounded
control validation and sample on the 3090, returning only token IDs. Two warmups
and six measured requests per group give 12 measurements per mode; all warmup
and measured outputs match exactly. Torch 2.6.0+cu124 / NCCL 2.21.5 run on both
hosts. NCCL Socket is explicitly selected on lab `enp6s18` and desktop `enp72s0`;
RDMA is disabled for this test. Per-worker hardware/transport metadata accompanies
`qwen2-network-token-ack-q8-abba.json`.

## Coordinator command admission without redundant readback

The coordinator now validates the encoded command it already owns instead of
reading the just-broadcast GPU header back to the CPU. The receiving worker
still reads and validates the transmitted header. Both workers still observe
collective admission before transferring activations; token completion and
cache ownership fences are unchanged.

A matched A2/3090 Q8 ABBA experiment (`--compare-admission`, 12 measured requests
per mode after warmup) compares the old readback with the local encoded record:

| Admission | Median first token | Median complete request | Median decode |
|---|---:|---:|---:|
| Coordinator GPU readback | 21.88 ms | 128.74 ms | 289.92 tokens/s |
| Coordinator local record | 21.83 ms | 127.53 ms | 293.89 tokens/s |

The observed decode gain is small (1.4%) and is from one matched experiment;
it is not a broad performance claim. Model, precision, context, prompt, output,
graphs, transport and hardware are unchanged between modes. All output tokens
match. Raw results and device identities are
`qwen2-network-admission-q8-abba.json` and its worker sidecars.

Both FP16 and Q8 independently pass all 82 numerical/token checks across 35
commands with the new admission path. Injecting a stale third command yields
final failure reports on both workers after exactly two completed commands,
before activation transfer. These reports are `qwen2-network-local-record-*.json`.
The local suite passes 398 tests, including peer rejection and admission
completion checks. No mocked test is used as GPU performance evidence.

Peer-only rules had automatic ten-minute expiry, opened 2026-10-06 at
00:42:26/28 UTC and removed early at 00:44:59/00:45:02 UTC. Original rules and
GPU allocations were verified restored; existing services were preserved.

## Bounded resident distributed sessions

`distributed-run --requests-file FILE` validates a coordinator-only JSON array
of 1–64 requests, bounded to 1 MiB, before either worker serves requests. One
pair retains weights and decode graphs for the session. Each completed request
uses acknowledged RELEASE before publishing its JSON result; the next prompt
gets a new slot generation. STOP terminates the pair after the last request.
Sessions are sequential and single-model, not distributed HTTP or model swaps.

The A2/A30 front-door verifier passes seven checks at both FP16 and Q8, retaining
the earlier independent single-request, EOS and invalid-token checks. The new
resident case runs prompt lengths 496, 31, 255, 31, 496, a chat ending on EOS,
and then another 31-token prompt in one worker pair. All reference continuations,
text, usage, ordering and finish reasons match; activation is charged only once.
An invalid eighth request is rejected before producing any output. Reports are
`qwen2-distributed-session-{fp16,q8}.json`.

The same seven-request Q8 session passes on the separate A2/3090 hosts. Only
the coordinator has the request file; neither prompt nor reference trace is
passed to the desktop command. Exact independent tokens and same-host text,
usage and finish reasons match. Both workers exit successfully. Reports are
`qwen2-network-session-q8.jsonl` and `qwen2-network-session-qualification.json`.
These are correctness/lifecycle timings, not warm performance measurements.
The local suite passes 401 tests.

The network session window automatically expired after ten minutes if left
alone. It opened 2026-10-06 at 00:51:45/49 UTC and was removed early at
00:53:03/06 UTC. Original firewall rules and GPU memory baselines were verified
unchanged. No existing GPU service was restarted.

### Complete repeated-request job cost

`scripts/bench_distributed_session.py` compares three sequential requests using
fresh A2/A30 workers for every request versus one resident pair for the whole
job. The pinned Qwen2.5-0.5B Q8 model, context 512, prompt lengths 31/255/31 and
eight output tokens per request are identical. Every result matches independent
reference tokens and usage; prepared-weight hashes agree throughout.

Fresh/resident/resident/fresh ordering includes one excluded warmup job and two
measured jobs per phase, giving four measured jobs per mode. Median complete-job
time falls from **46.17 to 15.82 seconds** (65.7% less elapsed time). Each measured
interval includes Python imports, worker launch, model activation, generation,
request release where applicable, and clean worker shutdown. OS file/compiler
caches are warm; GPU workers are fresh at each job boundary.

This quantifies amortized startup cost, not faster decode kernels, concurrent
throughput, HTTP performance or cross-host performance. Existing A30 services
remained running; no remote preparation or compilation job ran alongside the
measurements. GPU allocations returned to baseline after completion. Raw
per-request timings and all phases are in `qwen2-distributed-session-q8-abba.json`.

## Owned verification-job cleanup

Ordinary torchrun module workers create separate process groups. Killing only
the launcher can therefore leave workers behind. The same-host CLI verifier
and session benchmark now launch the actual CLI through torchrun `--run-path`,
whose multiprocessing workers retain the harness-owned group. Job deadlines
send TERM to that exact group, allow bounded graceful cleanup, then send KILL
and bound the final wait. External TERM of the harness also performs cleanup;
the caller's previous signal handler is restored afterward. No process-name
search or termination of other GPU services is used.

`scripts/verify_distributed_job.py` proves this with two real Torch 2.6 workers
that deliberately ignore TERM and INT. The 20-second deadline plus one-second
grace finishes in 21.06 seconds. Both workers' exclusive file locks are released,
and recorded worker/group identities match the signaled owned group. This is
CPU process-lifecycle evidence, not GPU performance proof. A separate real
seven-request A2/A30 Q8 session through the new launcher matches all previously
qualified tokens, text, usage and finish reasons and exits normally. Reports:
`distributed-job-deadline-final.json` and `qwen2-owned-session-smoke.json`.
The local suite passes 404 tests, including external harness termination.

The job-cost table above was measured using the direct-module benchmark harness
at commit `afa442e`. Its measurements are not refreshed timings for the later
run-path harness. Public distributed CLI commands still use ordinary torchrun;
this cleanup work is specific to owned local verification jobs and does not
add cross-host worker supervision or automatic service recovery.

## Native Q4 same-host partition

The qualified 12/12 Qwen2.5-0.5B same-host path now admits native `q4_0` alongside
FP16/Q8, retaining context <=512 and batch <=4. Layer sharding preserves packed
uint8 matrices and their FP16 scales without expanding weights. GGUF/AWQ,
paged KV, speculation and distributed Q4 CLI admission remain excluded.

Independent targets use upstream GGUF Q4 quantization/dequantization and
Transformers execution of the pinned source checkpoint. Both eager and graph
stages pass all 82 unchanged logit/token gates: worst relative RMS 0.005299,
maximum absolute error 0.046875, exact greedy agreement. Per-stage packed
weights/scales/KV are respectively 361796608/11182080/12582912 bytes on A2 and
157596416/19690496/12582912 bytes on A30. Scale tensors are checked on their
owning device. Raw stage reports are `qwen2-stages-q4-{eager,graphs}.json`.

Changed-weight A/B/A/B swaps plus recovery pass 410 independent checks. Graph,
weight, scale and KV addresses remain stable across compatible swaps. Injected
partial upload unpublishes the executor before a fresh rebuild, which passes
the original reference again (`qwen2-partition-rebind-q4.json`). Activation
timings in this lifecycle report are observations, not matched speedup evidence.

The normal HTTP gate passes 20 checks including independent continuations,
shrinking concurrent batches, streaming, cancellation/slot reuse, invalid
input recovery, output-GPU logprobs, seeded sampling/penalties, alias restore
and disk reload. Repeated unload returns to exactly 8519680 allocated bytes per
GPU, all identified as retained cuBLAS workspace, with no allocation growth.
It does not return to zero CUDA allocation. The allocation proof is
`qwen2-partition-http-q4-allocations.json`. All 405 local tests pass.

A separate injected last-stage decode failure also passes the full HTTP gate.
The damaged executor is reported failed, then the next request rebuilds from
host weights and matches the independent continuation. See
`qwen2-partition-http-q4-recovery.json`. This is recovery from an ordinary
execution exception, not proof of recovery from fatal CUDA/device faults.

The matched Q4 eager/graph/graph/eager experiment uses the same A2/A30 devices,
prepared weights, 255-token prompt, 32-token output and 512 context. Twelve
measured requests per mode follow excluded warmups. Median decode rises from
46.60 to 263.07 tokens/s, while complete-request time falls from 694.92 to
149.69 ms. First-token time is 29.02 ms eager versus 31.23 ms with stage graphs.
All measured tokens agree across modes. This proves the graph benefit at Q4;
it is not a Q4-versus-Q8 precision comparison, HTTP throughput measurement or
upstream-engine comparison. Raw results: `qwen2-stages-q4-graph-abba.json`.

## Native Q4 distributed CLI

The distributed path now admits Q4 through its own explicit precision guard,
separate from same-host partition admission. Before publication, the 3090
single-GPU path passes all 82 independent Q4 checks, and the A2/3090 network
pair passes those 82 checks across 35 commands. A stale third command produces
final failure reports on both workers after exactly two completed commands,
before activation transfer. See `qwen2-single-3090-q4-reference.json` and
`qwen2-network-control*q4-rank*.json`.

The actual Q4 CLI passes the full seven-check A2/A30 suite, including individual
reference prompts, EOS, invalid tokens, a seven-request resident session and
malformed later input rejected before any generation. The same seven-request
session also passes on separate A2/3090 hosts, with exact independent tokens
and matching text, usage, ordering, finish reasons and prepared-weight identity.
Only the coordinator receives the request file; both workers exit normally.
Reports are `qwen2-distributed-session-q4.json`, `qwen2-network-session-q4.jsonl`
and `qwen2-network-q4-session-qualification.json`. These runs include first-use
costs and are correctness evidence, not network performance measurements.
All 406 local tests pass.

The peer-only firewall window had automatic ten-minute expiry. It opened on
2026-10-06 at 01:21:44/47 UTC and was removed early at 01:24:53/57 UTC. Original
rules and GPU allocations were verified unchanged. No permanent firewall
changes or existing GPU-service restarts were made.

| Median network metric | Separate completion collective | Token reply completion |
| --- | ---: | ---: |
| First token | 20.88 ms | 20.76 ms |
| Full generation | 131.35 ms | 127.73 ms |
| Decode | 282.13 tokens/s | 290.07 tokens/s |

The measured gain is modest: 2.8% higher decode rate and 2.8% lower total latency.
This is a single sequential worker experiment with existing services preserved,
not HTTP throughput, a single-GPU comparison, or broad distributed serving parity.
The initial setup failure also showed that a surviving coordinator does not yet
immediately terminate when its peer fails before NCCL initialization; the test
owner explicitly stopped that coordinator within its external deadline. Worker
startup/failure supervision remains required for a serving implementation.

The peer-only firewall window began 2026-10-05 at 23:45:34 UTC (lab) and
23:45:38 UTC (desktop), with automatic 30-minute expiry. All three temporary
rules were removed early at 23:55:07/11 UTC. Lab rich rules returned to empty;
the desktop retained its original 10.42.0.0/16 rule. GPU allocations returned
to baseline: A2 0 MiB, A30 13,403 MiB, desktop 10,835 MiB. No permanent firewall
changes or existing GPU-service restarts were made.

## Reported preparation failures before NCCL

`StageSetup` uses torchrun's existing CPU rendezvous store to publish readiness
or a bounded exception-type label before NCCL initialization. Prefixes separate
agent runs, restart counts and caller-supplied setup sessions. Peers check status
between preparation steps, then wait for both ready states. This adds no work to
the token loop. Store operations use a five-second timeout; the preparation
status deadline defaults to 60 seconds. A blocked loader/compiler cannot be
interrupted by these checks, so the external process deadline remains required.

A real cross-host injection after desktop weight preparation produces explicit
final failure reports on both hosts, with zero inference commands. The A2 sees
`peer stage preparation failed: RuntimeError`; neither process requires manual
termination. A fresh A2/3090 pair then passes all 82 Q8 reference checks across
35 commands (`qwen2-network-setup-restart-rank*.json`). The same-host injected
probe was not used to claim propagation: its torchrun parent terminated the
sibling before that sibling could write a final report.

The benchmark entry point uses the same readiness guard. A short cross-host
integration run (`qwen2-network-setup-benchmark-smoke.json`) passes exact-token
agreement across its modes. It uses only one measured request per group and is
not a replacement for the earlier full performance experiment. The local suite
passes 391 tests.

The second peer-only window had automatic ten-minute expiry, opened on
2026-10-06 at 00:00:58/00:01:02 UTC and removed early at 00:04:58/00:05:02 UTC.
Original firewall rules and all three GPU memory baselines again remain intact.
Unreported crashes, interrupted preparation calls, in-flight network failures,
and production worker supervision are still distinct concerns; this change
specifically resolves the observed reported-preparation-failure path.

## Actual distributed generation command

`dllm distributed-run` now accepts a normal coordinator chat/raw prompt under
torchrun. Its output worker receives bounded commands and activations, selects
greedy tokens locally, and returns only token IDs. Both workers verify prepared
model identity and startup readiness. STOP completes before normal shutdown.
The command requires local snapshots, the qualified 12/12 Qwen2.5-0.5B geometry,
FP16 or Q8, context <=512 and one request. It does not add distributed HTTP
scheduling, model swaps, arbitrary sampling or automatic process restart.

The actual CLI passes five A2/A30 checks per precision: exact independent
reference tokens/text/usage for 31-, 255- and 496-token prompts with eight output
tokens; a normal chat prompt with EOS handling; and rejection of an invalid
vocabulary token. Every check launches and shuts down real worker processes.
Reports are `qwen2-distributed-cli-{fp16,q8}.json`; the local suite passes 396
tests. These are correctness/lifecycle runs, not performance comparisons.

The first chat invocation also demonstrated why post-activation timing must not
be called warm: lazy prompt-shape compilation can occur after model loading and
graph capture. The command reports `generation_first_token_seconds` for that
interval, while `first_token_seconds` includes activation. Repeated controlled
warm workloads remain necessary for performance comparisons.

The documented CLI invocation also passes on separate A2/3090 hosts: Q8 and FP16
normal chat both produce `4` and stop on EOS; a coordinator-only 255-token raw
prompt produces exactly the eight independent Q8 reference tokens, matching
text and usage. Both workers exit successfully in each run. The desktop command
receives neither a prompt argument nor a reference trace. Results and check
summary are `qwen2-network-cli-*.json`. These one-shot timings include first-use
costs and are correctness evidence, not warm performance comparisons.

The CLI test window automatically expired after ten minutes if left alone; it
was opened on 2026-10-06 at 00:19:50/54 UTC and removed early at 00:26:01/06 UTC.
Original firewall rules and GPU memory baselines were again verified unchanged.

## Single-GPU placement controls

The same pinned Qwen2.5-0.5B Q8 prepared weights, 512-token context, 255-token
prompt and 32-token greedy continuation were measured with one active GPU.
Each single-GPU path first passed all 82 independent reference checks. All
warmup and measured continuations then matched the distributed run exactly;
prepared-weight SHA256 identities also match. Each placement has 12 measured
requests after warmup, using native decode graphs.

| Active placement | Median first token | Median complete request | Median decode rate |
|---|---:|---:|---:|
| A2 alone | 29.39 ms | 157.04 ms | 243.13 tokens/s |
| 3090 alone | 7.69 ms | 49.51 ms | 741.77 tokens/s |
| A2/3090 network split, token completion | 20.76 ms | 127.73 ms | 290.07 tokens/s |

For this model and workload, the 3090 alone is the faster placement. The split
is a capacity option, not a general latency improvement. This comparison
deliberately changes active hardware, so it is not a same-hardware runtime
optimization claim. Single-GPU controls were run after the network experiment,
not interleaved with it. Existing GPU services remained running. These are
single-request warm measurements, not concurrent HTTP throughput, upstream
runtime comparisons or qualification of models requiring combined capacity.

Reproduce with `scripts/bench_single_stage_control.py`, supplying the pinned
local model, independent Q8 trace and the distributed benchmark report.
Raw reports: `qwen2-single-{a2,3090}-q8-{reference,placement}.json` and
`qwen2-network-token-ack-q8-abba.json`.


## Resident distributed HTTP

The experimental resident service passes FP16/Q8/Q4 A2/A30 HTTP qualification,
including queued concurrency, cancellation, acknowledged release and an idle
interval longer than the collective timeout. See the
[scope, reproduction and raw reports](http/README.md). This is a separate
greedy sequential queue; it does not add distributed model swapping or batching.

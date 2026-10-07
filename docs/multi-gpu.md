# Multi-GPU execution

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Start with the qualified same-host configuration below. Cross-host execution remains a separate experimental proof.

## Experimental two-GPU serving

`--layer-split 12` runs the first 12 decoder layers on the first visible GPU
and the remaining 12 on the second. Currently qualified for Qwen2.5-0.5B,
FP16, dllm Q8 or dllm Q4, context up to 512 and batch size up to four. It shares the
normal HTTP scheduler, streaming, cancellation and model residency controls.
Sampling stays on the output GPU; status includes both devices' allocations
and per-stage weights, scales and KV bytes.

```sh
CUDA_VISIBLE_DEVICES=GPU-first-uuid,GPU-second-uuid DLLM_QUANTIZATION=q8_0 \
  dllm serve --layer-split 12 --context 512 --batch-size 4 \
  --model small=/path/to/Qwen2.5-0.5B-Instruct
```

This is a sequential layer partition for same-host serving, not tensor
parallelism or a throughput pipeline. Paged KV, prefix caching, chunked prefill
and speculation are explicitly excluded in this mode. A partial stage failure
invalidates the executor; the next request rebuilds both stages from the host
cache. Cross-host A2/3090 execution has a separate experimental correctness
proof; it is not exposed by this serving option. RDMA remains unqualified.
See [multi-GPU evidence and limits](evidence/multi-gpu/README.md).

Set `DLLM_QUANTIZATION=q4_0` for the qualified native four-bit path. Packed
weights and scales stay with their owning stage; this is not GGUF or AWQ
partition support. Independent eager/graph, changed-weight swap, HTTP and
release checks are recorded in the [Q4 qualification](evidence/multi-gpu/README.md#native-q4-same-host-partition).
The separate distributed CLI also admits Q4 after its own network qualification.

## Compatible model swaps

Host-cached checkpoints with identical configuration, precision and weight/scale layouts reuse GPU buffers and captured graphs. This fast path is enabled by default; `DLLM_PARTITION_REBIND=0` disables it. Incompatible layouts rebuild normally. An interrupted upload removes the old residency and closes the executor before a fresh activation can occur.

Matched A2/A30 Q8 tests reduced median HTTP host restoration from 1.512 s to 0.211 s (about 7.2x). FP16 and Q8 each pass 410 independent checks across changed-weight swaps and recovery. These measurements apply to the qualified checkpoint pair, not arbitrary models or network transfers. See [complete measurements and limitations](evidence/multi-gpu/README.md#compatible-checkpoint-hot-swaps).

## Network and recovery boundaries

A2/desktop 3090 workers pass FP16, Q8 and Q4 independent checks using bounded
commands over NCCL sockets. The second worker does not read the reference trace;
sequence and slot-generation checks reject stale commands before activation
transfer. A matched Q8 network experiment measures 290 decode tokens/s with
token-return completion on its narrow 255-prompt/32-output workload.
See [network qualification and measurements](evidence/multi-gpu/README.md#command-driven-a23090-network-qualification).

Cross-host execution remains experimental. The resident HTTP service below uses
a separate greedy scheduler with up to four active slots.
Failed requests are not resumed. Reported preparation failures now propagate
through the CPU rendezvous store before NCCL starts; unreported crashes and
blocked preparation still need an outer process deadline. General worker
supervision remains incomplete.
No available test host exposes RDMA, so RDMA and GPUDirect RDMA remain unqualified.

## Run a prompt across two workers

`distributed-run` is an experimental single-request, greedy command launched
under torchrun. Prepare the same qualified Qwen2.5-0.5B-Instruct snapshot on both
workers with `dllm pull`; paths may differ, but prepared weight identities,
precision and context must agree. The command supports FP16 (`none`), Q8 and Q4,
context up to 512, and a 12/12 layer split.

For two GPUs on one host:

```sh
CUDA_VISIBLE_DEVICES=GPU-first-uuid,GPU-second-uuid OMP_NUM_THREADS=4 \
  torchrun --standalone --nproc-per-node=2 --module dllm distributed-run \
  /path/to/Qwen2.5-0.5B-Instruct --quantization q8_0 \
  --prompt 'What is 2 + 2? Answer with just the number.' --max-tokens 32
```

For two hosts, launch one rank on each using matching rendezvous settings.
Replace the UUIDs, coordinator IP, interface names and local model paths below.
The qualified network transport is NCCL Socket. Use trusted peers and a bounded
peer-only firewall window for the rendezvous port and NCCL socket connections.

Coordinator host:

```sh
CUDA_VISIBLE_DEVICES=GPU-coordinator-uuid OMP_NUM_THREADS=4 \
NCCL_NET=Socket NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=your-interface \
  torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr=10.0.0.10 --master-port=29571 --module dllm distributed-run \
  /local/model/snapshot --quantization q8_0 \
  --prompt 'Explain binary search briefly.' --max-tokens 64
```

Output-worker host:

```sh
CUDA_VISIBLE_DEVICES=GPU-output-worker-uuid OMP_NUM_THREADS=4 \
NCCL_NET=Socket NCCL_SOCKET_FAMILY=AF_INET NCCL_SOCKET_IFNAME=your-interface \
  torchrun --nnodes=2 --nproc-per-node=1 --node-rank=1 \
  --master-addr=10.0.0.10 --master-port=29571 --module dllm distributed-run \
  /other/local/model/snapshot --quantization q8_0
```

Only the coordinator reads the prompt and prints output. `--raw` selects raw
text; `--prompt-tokens-file FILE` reads a coordinator-local JSON token list.
`--json` emits one result containing text, token IDs, usage and timing.
`--ignore-eos` is available for fixed-length validation. The output worker
selects greedy tokens locally; full vocabulary logits do not cross the network.

Timing starts after Python imports and separates activation from generation.
Generation can still include first-use kernel compilation or transport setup. `first_token_seconds` includes activation;
`generation_first_token_seconds` starts after activation and is not necessarily
a warm measurement. Model files must already be local. Each command creates a
fresh pair; distributed model switching, HTTP scheduling, sampling controls,
stream reconnection and automatic worker restart are not implemented by this
command. The same-host HTTP server retains its separate residency/swap support.
Use an outer job deadline for crashes or blocked setup that cannot report status.

Choose a single GPU when it fits and measured latency is better. For the
qualified small Q8 workload, the 3090 alone measured 742 decode tokens/s,
versus 290 for the A2/3090 network split and 243 for the A2 alone. The split
does not automatically improve performance; see the
[placement controls and measurement limits](evidence/multi-gpu/README.md#single-gpu-placement-controls).

## Reuse a distributed pair for sequential requests

Pass `--requests-file requests.json` on the coordinator in place of `--prompt`
to keep weights and decode graphs resident across requests. The worker command
stays the same and does not need the request file. Example file:

```json
[
  {"prompt": "What is 2 + 2? Answer with just the number.", "max_tokens": 8},
  {"prompt": "Explain binary search briefly.", "max_tokens": 64},
  {"prompt": "The quick brown fox", "raw": true, "max_tokens": 16}
]
```

The file is a JSON array of 1–64 requests, at most 1 MiB. Each request has
either `prompt` text (at most 64 KiB) or a `tokens` list of valid raw vocabulary
IDs. Optional `max_tokens`, `ignore_eos` and text-only `raw` override the
coordinator's command-line defaults. Every request must fit the configured
context, including its requested output. All requests are validated before
generation starts, so a malformed later request cannot produce a partial run.

The coordinator emits one JSON line per completed request, with a zero-based
`request_index`, tokens, text, usage and timing. Both workers acknowledge cache
release before that line is published; subsequent requests get a new slot
generation. Weights and graphs stay resident, while prompts do not share chat
history or cached prefixes. A final acknowledged STOP ends the pair.

Only the first result includes activation time. Subsequent timings start at
their own request; `release_seconds` is reported separately from generation
and total time. First-use prompt-shape work can still occur on later requests.
These sessions are sequential file-driven jobs, not concurrent serving or
interactive chat. A failed job exits nonzero; previously emitted results do
not imply later requests succeeded. Retain the outer job deadline.

On the measured A2/A30 Q8 three-request workload, reusing the pair reduced
complete job time from 46.17 to 15.82 seconds, including startup and shutdown.
This saves repeated activation work; it is not a decode-speed improvement.
See the [matched job measurements](evidence/multi-gpu/README.md#complete-repeated-request-job-cost).

FP16, Q8 and Q4 are accepted by `--quantization` for individual requests and
sessions. Set the same precision on both workers. Q4 passes independent
A2/3090 protocol checks and an actual seven-request network session; see the
[distributed Q4 proof](evidence/multi-gpu/README.md#native-q4-distributed-cli).

## Packed prompt processing

The same-host layer-partition server packs FP16/Q8/Q4 prompts by default:
linear layers share work across prompt rows while attention and KV remain
isolated. Each temporary group stays within the configured context budget.
Set `DLLM_PARTITION_PACKED_PREFILL=0` for sequential prompt processing.
The resident distributed server uses the same policy when multiple slots are
configured. Set policy overrides identically on both workers; startup rejects
mismatches.

The qualified Q4 stages use adjacent-nibble body storage by default. Existing
device buffers are repacked without changing Q4 coefficients, checkpoint bytes,
or the output head. A2/A30 and A2/3090 pass the expanded numerical trace and real
HTTP checks; same-host rebinds preserve graph storage and fail closed on partial
uploads or repacking. `DLLM_Q4_STAGE_ADJACENT=0` restores legacy arithmetic and
its sequential-prefill default (the explicit prompt-packing flag still overrides
that default). Model status reports `weight_layout`. See the
[Q4 layout qualification and performance tradeoffs](evidence/quantization/cuda/q4-arithmetic/README.md).

Qualified adjacent Q4 stages also use selective runtime row bounds to avoid
compiling a new body kernel for every prompt length. Static bounds remain for
down-projection tile families where GPU measurements found regressions.
`DLLM_Q4_STAGE_RUNTIME_ROWS=0` restores entirely static bounds. This changes
compilation behavior, not the wire protocol or stored coefficients. Same-host
stage status exposes `runtime_row_bounds`; measured first-use savings and warm
A2/A30 and A2/3090 comparisons are in the Q4 evidence above.
The same stages reuse rotary query-output strides across prompt lengths by
default; `DLLM_Q4_STAGE_RUNTIME_ROPE=0` restores static stride selection.
Same-host stage status reports `runtime_rope_tokens`. This has separately
qualified first-use savings and small mixed warm network timing differences.

## Resident distributed HTTP service

`distributed-serve` keeps the qualified two-rank pair resident behind the
existing chat/completion API. Start it under torchrun on both workers, using
matching model snapshots, precision, context, batch capacity and dllm versions:

```bash
# Coordinator; select its GPU UUID and network interface as above.
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr=10.0.0.10 --master-port=29571 --module dllm distributed-serve \
  /local/model/snapshot --quantization q8_0 --batch-size 4 --alias a --port 8000

# Output worker; it exposes no HTTP listener.
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=1 \
  --master-addr=10.0.0.10 --master-port=29571 --module dllm distributed-serve \
  /other/local/model/snapshot --quantization q8_0 --batch-size 4
```

The coordinator listens on `127.0.0.1` only. Use its existing trusted local
access path or an SSH tunnel. It accepts `/v1/chat/completions` and
`/v1/completions`, including streaming and usage. `/v1/models/status` reports
resident precision, prepared model identity, queued/active work, completion
count and idle heartbeats. Model activation occurs before the listener opens.

Execution is greedy. `--batch-size` selects 1–4 resident request slots (default
1); set it identically on both workers. With multiple slots, active requests
share each decode pass. Completed or canceled slots are released and can admit
new requests while other slots continue decoding. FP16/Q8 prompts admitted in
the same iteration share packed prefill, bounded by one context's total rows.
Larger groups split in order; chunked prefill is not implemented. No admission
delay is added to wait for a fuller group.

`--queue-size` bounds waiting requests (default 8, maximum 64); `--queue-timeout` bounds their wait for
admission (default 120 seconds, maximum 300). Full/expired queues return 503.
Client disconnects cancel generation after the current transport operation;
both ranks acknowledge slot release before reuse. Successful final responses
are likewise withheld until release is acknowledged.

Idle heartbeats run every five seconds so the worker does not hit its
30-second collective timeout while waiting for a request. `--idle-timeout`
optionally shuts down an unused coordinator; zero, the default, keeps it
resident. A transport/execution failure poisons the pair and fails queued work;
start fresh workers to recover. Retain a process-owner deadline during tests.
There is no automatic worker restart or request replay.

The service retains the current Qwen2.5-0.5B geometry, 12/12 split, context
through 512 and FP16/Q8/Q4 support. Sampling changes, custom text stops,
logprobs, structured/tool output, embeddings, reranking, distributed model
registration/unload and hot swapping are rejected explicitly. The ordinary
same-host server continues to provide its separate broader feature set.
This service uses layer partitioning; it does not implement tensor parallelism.
Per-request `max_batch_size` reports the largest decode batch that actually
included that request. `prefill_batch_size` reports its actual prompt group;
status reports configured capacity and whether packing is enabled.

The control protocol is version 4: packed prompt lengths and ownership, acknowledged
idle heartbeats, and an explicit output-stage failure payload. An output-stage exception can use the existing
token transfer to fail the session when CUDA and the transport remain usable;
this adds no successful-path collective. CUDA context corruption, broken
transport or process death may still abort workers and need the outer process
owner to terminate the job. The service does not promise graceful HTTP errors
for those fatal failures.
Both workers must use the same dllm revision. Startup also checks a fixed-size
digest of prepared weights, protocol version, context, batch capacity and packing policy before
opening the command channel. See the
[HTTP qualification evidence](evidence/multi-gpu/README.md#resident-distributed-http).

## Modern hybrid execution boundary (implementation only)

`HybridExecutor.forward_layers` now executes a validated contiguous range using
global layer indices and unchanged attention positions. Embedding and the final
head are separate methods; the range returns every hidden position and does not
commit request cursors. The ordinary single-device path uses these same methods.
CPU regressions compose every split in a four-layer hybrid fixture and compare
full prefill, subsequent decode, chunk offsets and interleaved slots. Unselected
layers' KV, convolution and recurrent state remain untouched. This is shared
execution code, not multi-device allocation, transport or numerical qualification.

`HybridLayerStage` now builds a source-weight view and allocates only the chosen
layers' KV, recurrent and convolution state. Global indices remain unchanged;
only the first stage owns embedding, only the last owns norm/head. Per-stage
admission counts that stage's weight representation and state plus workspace.
Tied embedding/head storage is counted on both stages when they live separately.
The current building block rejects quantization, graphs and host layer streaming.
It exposes the layer boundary, and rejects ordinary executor calls that would
commit a stage cursor before a coordinator completes the whole request. CPU
composition and exact allocation-count tests pass; device transfer, coordinated
failure handling and serving admission remain separate work.

The in-process `HybridPipeline` coordinator composes preconstructed stages with
blocking hidden-state copies. It accepts chunked prefills and reordered decode
slots and commits all stage/request cursors only after every stage synchronizes.
A failure or cancellation after device work begins invalidates the whole
pipeline; cancellation before work leaves it usable. Stages cannot belong to two
coordinators. The same CPU source identity, contiguous ranges, precision and
request limits must agree at construction. Closing synchronizes all participants
before releasing owned tensors, and preserves ownership if synchronization fails.

Stage admission additionally reserves two FP32 hidden-activation buffers at the
configured context/batch bound and the final stage's eager FP32 head temporary.
Status reports owned weight/state bytes and reserved workspaces separately from
allocator telemetry. CPU regression evidence covers composition, reordered
batches, interleaved slots, cursor commit, cancellation, failure and cleanup.
This remains an internal unquantized eager building block: no CLI/HTTP distributed
admission, remote transport, real-device transfer or performance claim is added.

## CPU host-staged transport foundation

`dllm.host_tensor_channel.HostTensorChannel` exchanges contiguous CPU tensors
through an already-connected stream socket supplied by its owner. It creates no
listener and is not wired into model workers or public serving admission. Peers
must agree on a 32-byte model/session identity, protocol version, opposite ranks
and native byte order before exchanging tensors. Identity agreement is not peer
authentication or encryption; connection security belongs to the owner.

Frames use fixed binary metadata, per-direction sequence numbers and a negotiated
cap of at most 64 MiB. The receiver supplies the exact expected shape and dtype;
metadata mismatch is rejected before payload allocation. FP32, FP16, BF16, int64,
int32 and uint8 have explicit wire codes; no pickle or Python-object transport is
used. Each operation has one absolute deadline across all partial sends/reads.
Protocol, EOF, deadline and I/O failures close the owned socket and poison the
channel. Concurrent close interrupts blocked operations; short socket polling
intervals preserve cancellation responsiveness without extending the operation
deadline. Local invalid send/receive layouts are rejected before any wire write.
Received tensors own independent CPU payload storage. Senders must keep their
input tensors unchanged until send returns. A successful frame transfer is not
an acknowledgement that a remote model stage committed its state.

Loopback regression tests cover byte-exact types, identity mismatch, malformed
headers before payload reads, truncated payloads, absolute deadlines and close
cancellation. Remote worker lifecycle, model-state acknowledgement, secure socket
provisioning, GPU copies, cross-vendor execution, RDMA and throughput remain
unqualified. No model-serving or hardware-support claim follows from these tests.

`dllm.host_stage_channel.HostStageChannel` binds the owned host tensor transport
to fixed stage command and completion records. Its setup exchange checks epoch,
context, slot count and vocabulary after the caller-supplied model/session
identity agrees. Both peers admit a command before the caller may transfer
activations. Execution completion transfers checked sampled int64 IDs and
exchanges a phase/sequence-bound acknowledgement before the coordinator commits
its slot cursor. Release, ping and stop use the same acknowledgement boundary.
The owner must finish CPU work before acknowledging and abort on execution errors.

Local TCP tests cover prefill, decode, packed prefill, slot release/reuse, stop,
receiver ownership rejection, corrupted acknowledgements/results, deadlines and
prompt close cancellation. This retains `StageState`'s four-slot bound and default 512 context limit;
explicit larger host control contexts are described below. It is
transport/lifecycle groundwork, not modern-model qualification. There is no remote model worker,
listener, authenticated provisioning or GPU fence in this component. Each frame
uses an absolute transport deadline; each method has a fixed number of frames.

Completion is not a durable distributed transaction: a peer can send its last
acknowledgement before learning that the other side disconnected or rejected it.
Peers may briefly have different local commit positions. Failure closes and
poisons the epoch when observed; uncertain commands are never replayed and no
cache recovery is attempted. A fresh owned session is required after failure.

`HostHybridWorker` now composes two prepared `HybridLayerStage` instances with
the host control/tensor channel. Each worker owns only its contiguous layer
range and local KV/recurrent/convolution state. Setup checks matching normalized
configuration/FP32 precision and complete nonoverlapping ranges; the transport
owner still supplies the shared weight/session identity. Ordinary/packed prefill,
reordered or shrinking decode, release/reuse, ping and stop are supported within
four slots and the explicitly admitted control context (512 by default). Only sampled token
IDs return from the final stage. Input token/slot/frame checks precede commands.

The implementation requires prepared CPU FP32 stages. Local TCP tests compare
sampled output and all retained layer state with the independent single-executor
control, including interleaved slots and reuse. Release clears the selected local
cache tensors before acknowledgement. Stop closes owned stage resources before
acknowledgement. A stage/transport failure invalidates the session and closes
both workers when observed. Close interrupts blocked socket reads; failed stage
cleanup keeps ownership for retry and does not replace a primary execution error.

This supplies prepared worker composition, not a listener, remote provisioning,
HTTP registration, quantized execution, GPU transport, practical long context or
network performance qualification. A caller must explicitly own connected trusted
sockets and prepared stages. It does not build a full executor on each worker.

Protocol version 5 adds an owned `PREFILL_CHUNK` command. A fresh chunk requires
an empty slot and its next generation; continuing chunks require the exact live
generation and current offset. Commit advances by the chunk length only after
result/completion acknowledgement. Stale offsets, live-slot restarts and context
overflow fail. The version change prevents old peers from silently accepting the
new semantics; both participants must use the same protocol version.

`StageState(..., context_limit=...)` permits an explicit control bound up to
1,048,576 while its default remains 512. This does not change legacy/GPU model
admission. `HostHybridWorker.prefill_chunk(tokens, slot, start)` executes one
bounded frame and returns that chunk's sampled continuation. Callers keep
appending prompt chunks before switching to decode. The prepared stage's actual
context/state budget must agree with the control session. Oversized unchunked
frames fail before a command is sent; no prompt truncation or hidden downgrade
occurs. Tests cover 32K control counters without tensor allocation, and actual
synthetic 513-token CPU arithmetic/state parity through 128-token loopback frames.
Neither result qualifies practical checkpoint context, quality or hardware speed.

### Authenticated connected host sockets

`host_tls.client_context` and `server_context` load an explicit trust CA and
local certificate/key, require peer certificates and TLS 1.3, and negotiate
`dllm-host/1`. `wrap_peer` takes an already-connected TCP socket, a context, the
exact expected SHA-256 DER certificate pin and (on the client) a server hostname.
It checks hostname/trust, protocol and pin before returning the owned TLS socket
for `HostTensorChannel`. Model/session digest agreement and stage setup still
follow TLS. There is no plaintext fallback or listener/connector deployment.

Local option errors leave the socket with its caller; wrapping/handshake errors
close the owned descriptor. The handshake deadline is bounded to at most thirty
seconds. Configured contexts must remain immutable during use. Certificate
issuance, protected private-key storage, peer-pin distribution/rotation and
listener/service provisioning remain operator responsibilities and unimplemented
deployment gates. Published credentials under `tests/fixtures/host_tls` are
synthetic test-only keys and must never be used for a deployed service.

Actual loopback tests cover tensor/stage frames, certificate/hostname/pin/ALPN
rejection, stalled handshake cleanup and close interruption. TLS 1.3 may surface
a rejected client certificate at the client's first application I/O; the required
framed session handshake must succeed before any stage is admitted. No remote
hardware, network throughput or GPU qualification is claimed.

### Prepared CPU quantized stage reference

`dllm.cpu_hybrid_stage.CpuHybridStage` accepts explicitly prepared CPU
Qwen3.5 text tensors in floating, canonical Q4_0 or Q8_0 form. Each stage owns
only its layer caches and selected weight/scale bindings. Packed projections
remain tiled; floating controls use FP32. Constructor limits admit retained
weights, cache/rotary tables and conversion scratch before allocation, while
execution separately bounds workspace and outputs. A failure after execution
starts poisons the stage; close remains available. Coordinators own cursors.

The host worker accepts these explicit CPU stages and includes quantization in
its setup contract. Fixture tests compare split stages, chunked prefill,
reordered decode and actual loopback token exchange with decoded FP32 controls.
This is a CPU reference building block, not accelerator admission, trained-model
quality, practical-context or network performance qualification. Existing GPU
stage restrictions remain in force.

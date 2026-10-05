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

Failure recovery, serving integration, RDMA, and GPUDirect RDMA remain
unqualified. The model partition experiments below are separate evidence;
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

The stage implementation lives under `scripts/experiments` and is not registered
for serving. It executes stages sequentially and copies their activations. It
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

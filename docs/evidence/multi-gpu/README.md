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

Model partitioning, failure recovery, serving integration, RDMA, and GPUDirect
RDMA remain unqualified. A successful NCCL Socket test establishes none of these.

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

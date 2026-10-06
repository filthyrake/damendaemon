# Multi-GPU execution

[Documentation](README.md) · [Project](../README.md) · [Support matrix](support.md)

Start with the qualified same-host configuration below. Cross-host execution remains a separate experimental proof.

## Experimental two-GPU serving

`--layer-split 12` runs the first 12 decoder layers on the first visible GPU
and the remaining 12 on the second. Currently qualified for Qwen2.5-0.5B,
FP16 or dllm Q8, context up to 512 and batch size up to four. It shares the
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

## Compatible model swaps

Host-cached checkpoints with identical configuration, precision and weight/scale layouts reuse GPU buffers and captured graphs. This fast path is enabled by default; `DLLM_PARTITION_REBIND=0` disables it. Incompatible layouts rebuild normally. An interrupted upload removes the old residency and closes the executor before a fresh activation can occur.

Matched A2/A30 Q8 tests reduced median HTTP host restoration from 1.512 s to 0.211 s (about 7.2x). FP16 and Q8 each pass 410 independent checks across changed-weight swaps and recovery. These measurements apply to the qualified checkpoint pair, not arbitrary models or network transfers. See [complete measurements and limitations](evidence/multi-gpu/README.md#compatible-checkpoint-hot-swaps).

## Network and recovery boundaries

A2/desktop 3090 workers pass FP16 and Q8 independent checks using bounded
commands over NCCL sockets. The second worker does not read the reference trace;
sequence and slot-generation checks reject stale commands before activation
transfer. A matched Q8 network experiment measures 290 decode tokens/s with
token-return completion on its narrow 255-prompt/32-output workload.
See [network qualification and measurements](evidence/multi-gpu/README.md#command-driven-a23090-network-qualification).

This remains an experimental worker harness, not an HTTP worker service.
Failed requests are not resumed. Reported preparation failures now propagate
through the CPU rendezvous store before NCCL starts; unreported crashes and
blocked preparation still need an outer process deadline. General worker
supervision remains incomplete.
No available test host exposes RDMA, so RDMA and GPUDirect RDMA remain unqualified.

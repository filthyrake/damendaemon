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

Cross-host execution remains experimental and is not connected to the HTTP scheduler.
Failed requests are not resumed. Reported preparation failures now propagate
through the CPU rendezvous store before NCCL starts; unreported crashes and
blocked preparation still need an outer process deadline. General worker
supervision remains incomplete.
No available test host exposes RDMA, so RDMA and GPUDirect RDMA remain unqualified.

## Run a prompt across two workers

`distributed-run` is an experimental single-request, greedy command launched
under torchrun. Prepare the same qualified Qwen2.5-0.5B-Instruct snapshot on both
workers with `dllm pull`; paths may differ, but prepared weight identities,
precision and context must agree. The command supports FP16 (`none`) and Q8,
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

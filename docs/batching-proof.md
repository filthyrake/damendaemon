# Fused kernels and continuous batching

Measured 2026-10-03 on the authorized NVIDIA A2, FP16 Qwen2.5. This extends the
[first runtime proof](runtime-proof.md). The A30 workload was left running.
Both immutable model revisions, driver, and hardware are the same as that proof.
dllm uses PyTorch 2.6.0+cu124 and Triton 3.2.0; the independent comparison stack
uses vLLM 0.30.0 with its resolved PyTorch 2.13.0. These are different software
stacks on the same GPU, not a controlled comparison of one isolated kernel.

## Why the first executor lost

A CUDA profile identified 1,364 kernel launches per generated token. Separate
casts, RMS normalization, RoPE, cache writes, activation, and general tensor
attention added work around the weight matrix operations. CUDA graphs removed
Python dispatch but did not fuse those operations. The service also serialized
requests, repeatedly reading the same weights for each user.

dllm now implements fused Triton RMSNorm, SwiGLU, RoPE/cache writes, and
live-prefix attention. FP32 reductions and explicit FP16 rounding preserve the
reference calculation. Attention partitions long contexts into 256-position
blocks and merges them with log-sum-exp weights, skipping blocks past the live
position. Weight matrix operations still use PyTorch/cuBLAS.

The same profile now counts 274 kernels/token, with summed GPU kernel time
falling from 9.14 to 6.63 ms/token. Profiling adds overhead; these timings explain
costs, while the unprofiled runs below establish speed.

Raw profiles: [before](evidence/profile-before.json),
[after](evidence/profile-after.json).

## Scheduling and prompt processing

One worker owns the GPU. It continuously admits requests into independent KV
slots, captures a decode graph per batch size, and releases completed/cancelled
slots between steps. A different model at the head of the FIFO stops admission
to the current model group; that group drains before switching. A slow consumer
has a bounded eight-event queue and eventually times out rather than consuming
unbounded memory. Closing the runtime wakes waiting consumers.

Available prompts share packed linear layers, even when their lengths differ;
causal attention remains separate for every sequence. There is no artificial
batch-formation delay. Decode metadata uses a reusable pinned buffer and one
host-to-device transfer. A CUDA event prevents overwriting that buffer before
its prior DMA completes. Normal serving defaults to four slots; the CLI accepts
one through eight.

A worker profile also found eager prompt processing and small metadata transfers
were material costs beyond decode. Intermediate implementations and their
measurements are retained, including slower results. Packing prompts does not
eliminate eager dispatch or long-prompt stalls: prefill is still unchunked.

## Correctness and recovery

The original numerical gate is unchanged: exact greedy selections, relative RMS
logit error below 1%, and maximum absolute logit error below 0.5 against a separate
Transformers 4.51.3 Qwen2 implementation with FP16 SDPA.

- Both fused single-sequence models passed 96 positions each across eager and
  graph execution. Maximum relative errors were 0.598% and 0.611%.
- Both batched models passed 48 positions each with different prompt lengths,
  reordered/non-contiguous slots, packed prefill, and a reused slot after changing
  batch size. The PyTorch fallback passed the same batched check on 0.5B.
- Fused kernel checks cover normalization, activation, attention split boundaries,
  and reordered slots at positions 0, 256, 512, and 1023. Unused future KV entries
  are poisoned with NaNs to check that they cannot affect output.
- The real CLI/API passes A→B→A→B→A stability with one and two resident models,
  matching streaming/non-streaming output, four simultaneous same-model requests,
  disconnects, mixed-model requests, invalid context, and failed-load recovery.
- A cold compiler-cache test deliberately supplies a missing compiler. Automatic
  selection completes through the PyTorch fallback; explicit Triton selection
  fails rather than silently benchmarking the slower backend.
- 33 local tests cover API contracts and lifecycle failures, including bounded
  admission, queue timeout, slow-consumer timeout, cancellation, shutdown,
  cross-model fairness, prompt errors, and decode errors. They are not GPU proof.

Evidence: [fused 0.5B](evidence/verify-fused-a.json),
[fused 1.5B](evidence/verify-fused-b.json),
[batched 0.5B](evidence/verify-packed-a.json),
[batched 1.5B](evidence/verify-packed-b.json),
[PyTorch fallback](evidence/verify-batch-torch.json),
[kernel boundaries](evidence/kernels-final.json),
[compiler fallback](evidence/backend-fallback.json),
[real API](evidence/service-batched-final.json).

## Performance measurement

The HTTP workload uses the exact 0.5B snapshot, a 43-token chat prompt, 64 greedy
output tokens, context capacity 256, four allowed active sequences, four CPU
threads, and no prefix caching. Engines run sequentially on the A2. The client
runs six rounds at each concurrency and excludes the first warmup round. Usage
must report 64 output tokens and the stream must finish correctly. Aggregate
throughput is all output tokens divided by round wall time, including prefill
and HTTP delivery. Per-request post-first-output rate is a different metric.

| Engine | Concurrent requests | Aggregate output tokens/s | Per-request post-first tokens/s | First output (ms) |
| --- | --- | --- | --- | --- |
| dllm | 1 | 133.3 | 138.8 | 26.0 |
| vllm | 1 | 138.9 | 149.1 | 38.1 |
| dllm | 2 | 225.6 | 121.2 | 41.5 |
| vllm | 2 | 247.3 | 136.5 | 52.3 |
| dllm | 4 | 432.7 | 117.3 | 60.2 |
| vllm | 4 | 496.9 | 138.1 | 56.8 |

Raw final trials: [dllm, C1](evidence/compare-dllm-batched-c1.json), [dllm, C2](evidence/compare-dllm-batched-c2.json), [dllm, C4](evidence/compare-dllm-batched-c4.json), [vllm, C1](evidence/compare-vllm-batched-c1.json), [vllm, C2](evidence/compare-vllm-batched-c2.json), [vllm, C4](evidence/compare-vllm-batched-c4.json).

All final generated response texts matched across the two engines at each concurrency.

The final single-request aggregate rate is about 4% below vLLM, while its
post-first-output rate is about 7% below. At four requests, aggregate throughput
is about 13% below vLLM. The original serialized dllm post-first rate of 106.1
rose to 138.8 tokens/s, approximately a 31% improvement. The final executor-only
batched test at context 512 measured approximately 145 / 261 / 530 aggregate
decode tokens/s for 0.5B at batch sizes 1 / 2 / 4, and 50 / 96 / 188 for 1.5B.
Those executor numbers exclude prefill and HTTP and must not be substituted
for the API comparison.

These short shared-VM measurements do not establish broad parity, tail latency,
long-context performance, or superiority across models. The original 0.5B
serialized API delivered about 106 tokens/s after first output; the original
llama.cpp baseline delivered about 66. That llama.cpp run is historical and was
not repeated for concurrent throughput.

The intermediate `prepacked-compare-*.json` and `packed-tensor-compare-*.json`
files preserve earlier versions. They must not be mixed with the final trials
or interpreted as measurements of the final source.

## Model switching and limits

GPU-resident models retain their graphs. An evicted model restores host weights
and recaptures each batch-size graph. This costs more than the original single-
graph executor; throughput improvements do not imply faster cold restoration.
The final API proof measured host activation of 0.99 s for 0.5B and 1.61 s for
1.5B. Resident request-to-first-token latency was 20–21 ms and 26 ms respectively.
The raw API evidence reports activation and request-to-first-token separately.
"Disk" means no process-local weight cache; the OS cache was not flushed.

Remaining costs include eager prompt processing, GPU matrix-vector/matrix
operations, token delivery, and graph reconstruction on model restoration.
There is no paged KV, prefix cache, quantization, speculative decoding, or
chunked prefill. Fixed KV slots reserve capacity even for short requests.
Architectures beyond dense Qwen2 and non-greedy sampling remain unsupported.
The loopback server is a development service, not a public gateway.

## Reproduce

Install the README runtime/validation extras, CUDA PyTorch, a C compiler, and
matching Python headers. Select exactly one GPU UUID. Require the fast backend
for performance runs:

```sh
export DLLM_KERNELS=triton
python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/verify_kernels.py
PYTHONPATH=. python scripts/verify_qwen2.py /path/to/model
PYTHONPATH=. python scripts/verify_batch.py /path/to/model
DLLM_KERNELS=torch PYTHONPATH=. python scripts/verify_batch.py /path/to/model
PYTHONPATH=. python scripts/verify_backend.py /path/to/model
PYTHONPATH=. python scripts/verify_service.py /path/to/0.5b /path/to/1.5b
PYTHONPATH=. python scripts/profile_decode.py /path/to/model
PYTHONPATH=. python scripts/profile_scheduler.py /path/to/model
PYTHONPATH=. python scripts/compare_llama.py http://127.0.0.1:8088/v1 --engine dllm --concurrency 4
```

`scripts/compare_a2.py` reproduces the lab's sequential dllm/vLLM comparison
using its existing isolated virtual environments and `model-a.json`. It refuses
any GPU UUID other than the authorized A2 and only terminates its own process
groups. vLLM remains a separate benchmark dependency; dllm never imports it.

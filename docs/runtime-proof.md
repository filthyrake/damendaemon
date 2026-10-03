# First independent runtime proof

Measured on 2026-10-03, NVIDIA A2 (15 GiB usable), compute capability 8.6.
All test GPU processes selected the A2 by UUID. The A30 continued its existing
workload. Temporary dllm and baseline servers were stopped after their tests;
test environments, downloads, and logs remain in `~/dllm-dev/` on the lab VM.

## Model identities and runtime

| Model | Immutable Hugging Face revision |
| --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

dllm: FP16 weights and KV, PyTorch 2.6.0+cu124, tokenizer 0.21.1, safetensors
0.5.3. Context capacity 256 for these measurements. Python 3.11. No quantization,
prefix caching, speculation, batching, or offloaded inference. Host CPU threads
were capped at four for the API tests. These are shared-VM measurements; no GPU
clock or power settings were changed.

## Correctness

`scripts/verify_qwen2.py` compares dllm against Transformers 4.51.3's separate
Qwen2 implementation with FP16 and SDPA. It checks chat-template token equality,
then compares the full vocabulary logits and greedy token at each of 16
teacher-forced positions for three prompts, with and without CUDA graphs.
Both model sizes passed: 192/192 greedy selections matched. Reusing the same
executor across prompts also exercises isolation from stale KV entries.

The gate requires exact greedy selections, maximum relative RMS logit error
below 1%, and maximum absolute error below 0.5. Observed maximum relative errors
were 0.651% (0.5B) and 0.968% (1.5B), with maximum absolute errors 0.221 and 0.160.
These are finite-precision comparisons, not bitwise equality. An initial fixed
absolute/mean cutoff was too strict for the observed FP16 paths and failed;
complete diagnostic runs and an unfused-projection check preceded adoption of
the relative-error-plus-token gate. No greedy mismatch was waived.

The correctness loop intentionally continues after EOS to exercise 16 cache
updates; text after EOS in the evidence is not a normal service completion.
This is a focused implementation check, not a broad model-quality evaluation
or a guarantee over all prompts and context lengths.

Raw evidence: [0.5B](evidence/qwen-0.5b.json), [1.5B](evidence/qwen-1.5b.json).

## Executor performance

Each trial prefills the same short prompt and generates 64 tokens, ignoring EOS
for fixed work. The decode rate covers the remaining 63 steps after the first
token. It excludes model loading and graph capture. Three trials per mode.

| Model | Eager decode | CUDA graph decode |
| --- | --- | --- |
| 0.5B | 37.1–37.8 tokens/s | 109.9–110.8 tokens/s |
| 1.5B | 30.5–32.0 tokens/s | 43.0–43.3 tokens/s |

## Real API and model switching

`scripts/verify_service.py` launches the actual `python -m dllm serve` command,
then sends HTTP requests. It verifies A -> B -> A -> B -> A output stability,
streaming/non-streaming agreement, usage counts, two overlapping requests,
disconnect recovery, oversized-request rejection, and recovery after an invalid
model path. It runs once with one GPU slot and once with two.

| State | Observed time |
| --- | --- |
| Host-cached 0.5B activation after eviction | 0.63–0.85 s |
| Host-cached 1.5B activation after eviction | 1.18 s |
| Already resident 0.5B, request to first token | 23–27 ms |
| Already resident 1.5B, request to first token | 31 ms |

Activation includes eviction, weight transfer, allocation, and graph recapture;
it excludes subsequent prompt prefill. Resident first-token time includes
prefill. Selecting an already resident executor itself is only a few microseconds,
but that is not the end-user response latency. "Disk" in the raw data means no
process-local host cache: OS storage caches were not flushed, so it is not a
cold-storage benchmark. Small sample counts do not establish tail latency.

Raw evidence: [service proof](evidence/service.json).

The final source passed a [second complete API run](evidence/service-final.json).
A separate [real GPU recovery check](evidence/recovery.json) verifies normal EOS
stopping, host-weight-budget rejection before eviction of the ready model, and
direct generator cancellation followed by a successful request. The installed
`dllm run` command was also exercised with the pinned repository name and default
1024-token context; it returned `4` and stopped after two generated tokens.

## Matching llama.cpp HTTP baseline

Converted the exact 0.5B snapshot to F16 using the unchanged local llama.cpp
converter at `836d57176dc699a726c55418e4f96b8ca628e1bf`. Used the existing CUDA
server image with digest
`sha256:f90b9de8baf5244c2c43f9594d7522424470f0d472323fe10405eceae714405b`.
All layers on A2, flash attention enabled, one slot, 256 context, Jinja template,
default FP16 KV. The server's CPU thread setting was 16; dllm used four.

`scripts/compare_llama.py` sends the same 43-token chat prompt and asks for 64
greedy output tokens. Both report 43 input and 64 output tokens. llama.cpp prompt
caching is disabled per request, and its usage confirms zero cached tokens.
The GPU engines run sequentially. One warmup request is excluded; five measured
requests remain. Both produce coherent responses, but their generated token
sequences differ, so this is equal-length work, not identical continuation tokens.

| Engine | Median total HTTP time | Median first output | Median post-first output rate |
| --- | --- | --- | --- |
| dllm | 0.627 s | 31.0 ms | 106.1 tokens/s |
| llama.cpp image above | 0.995 s | 33.7 ms | 65.7 tokens/s |

The rate is `(completion_tokens - 1) / (request_duration - first_output_latency)`.
It includes network delivery and tokenizer work and is not GPU-only decode
instrumentation. It must not be interpreted as token-level p95 latency.
The ratio is approximately 1.61x for this specific short, single-request FP16
workload. It does not establish superiority for quantized models, larger models,
long prompts, concurrent throughput, or different hardware.

Raw evidence: [dllm](evidence/compare-dllm.json), [llama.cpp](evidence/compare-llama.json).

## Current vLLM baseline

The same HTTP client/workload was also run against vLLM 0.30.0, installed in a
separate benchmark-only virtual environment. Its resolved PyTorch version was
2.13.0. Configuration: the same 0.5B snapshot, `--dtype half`,
`--max-model-len 256`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.5`,
`--no-enable-prefix-caching`, and `--generation-config vllm`. Default compilation
and CUDA graphs remained enabled. CPU threads were capped at four. No vLLM code
or runtime dependency is used by dllm.

vLLM required Python development headers for its JIT compilation. Headers were
extracted into the benchmark directory from `python3.11-devel-3.11.13-10.el9_8.x86_64`
and supplied via `C_INCLUDE_PATH`; no system package was installed. The virtual
environment's `bin` directory was placed on PATH for its Ninja compiler driver.
Compilation and graph capture were allowed to finish before measurement.

| Engine | Median total HTTP time | Median first output | Median post-first output rate |
| --- | --- | --- | --- |
| vLLM 0.30.0 | 0.461 s | 40.4 ms | 149.6 tokens/s |
| dllm | 0.627 s | 31.0 ms | 106.1 tokens/s |

The completed 64-token response text matched dllm's response in all six trials.
dllm is approximately 29% below vLLM's decode delivery rate in this test; there
is no performance-parity claim. The lower measured first-output latency is a
small-sample observation, not a tail-latency claim. Concurrent-throughput benchmarks
would not be meaningful parity tests yet because dllm serializes requests.
Different engine dependency/toolkit versions are part of these tested stacks.

Raw evidence: [vLLM](evidence/compare-vllm.json).

## Reproduce the implementation gates

Install `.[runtime,validation]` using the CUDA installation instructions in the
README, select one GPU UUID, and use the immutable snapshot directories:

```sh
python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/verify_qwen2.py /path/to/0.5b/snapshot
PYTHONPATH=. python scripts/verify_qwen2.py /path/to/1.5b/snapshot
PYTHONPATH=. python scripts/verify_service.py /path/to/0.5b/snapshot /path/to/1.5b/snapshot
PYTHONPATH=. python scripts/verify_recovery.py /path/to/0.5b/snapshot /path/to/1.5b/snapshot
```

Unit tests use fake weights/executors only for lifecycle faults; they do not
provide GPU performance evidence. The scripts above perform the real GPU proof.

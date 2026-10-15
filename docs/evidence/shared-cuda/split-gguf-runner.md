# Bounded split-GGUF runner (#217 / parent #43)

Implemented locally; real CUDA qualification is still pending. CPU fixtures and
loopback tests do not qualify a real model, GPU, performance, or production SLO.

## Prepare without a GPU

Supply an existing single-file checkpoint and a standard numbered split copy
of the same checkpoint. The runner does not download or modify either input.
A new output directory is required for each attempt.

```sh
python -m scripts.experiments.split_gguf --prepare \
  /absolute/model.gguf /absolute/model-00001-of-00002.gguf /absolute/new-evidence
```

Preparation validates supported small dense Qwen2/Qwen3/Llama architecture,
complete shard membership, identical tensor names/shapes/types/packed bytes,
config, tokenizer, template and special tokens. It records SHA-256 for every
file and packed tensor; file hashes are rechecked before and after each worker.
Host weights must be at most 1 GiB and checkpoint files at most 2 GiB. Preparation
uses CPU only. Defaults are context 128 and four greedy output tokens for a
fixed short raw prompt; the Python preparation API accepts context 32–128 and
one to eight tokens. There is no context/precision auto-reduction to hide OOM.

## Run alongside an active service

Use fresh PID, URL and GPU UUID values obtained from read-only host inventory.
This example is a template, not an instruction to reuse stale service identity.

```sh
python -m scripts.experiments.guard_shared_cuda \
  --allow-busy --gpu GPU-EXPLICIT-UUID \
  --service http://INCUMBENT-IP:8080 --incumbent-pid INCUMBENT_PID \
  --timeout 120 --report /absolute/new-guard-report.json -- \
  python -m scripts.experiments.split_gguf --run /absolute/new-evidence
```

`--allow-busy` admits healthy active slots while retaining process identity,
telemetry, GPU/host memory reserve and temperature checks. It does not partition
compute, guarantee latency, or alter the incumbent service. Omitting it keeps
idle-only admission. An environment marker prevents accidental direct CLI
execution without the guard; it is not a security boundary.

The run has four strictly sequential fresh processes: single trace, split trace,
single HTTP, split HTTP. Each sets a 1536 MiB PyTorch allocator cap before model
allocation, disables CUDA graphs, uses one model resident at a time, and checks
allocator cleanup. CUDA driver/context and external-library allocations are
outside that cap; the external GPU free-memory reserve still matters. No MPS,
MIG, driver, fan or power configuration is changed. Each process exits before
its successor starts. No extra worker process groups are created.

Numerical acceptance requires finite, byte-identical FP32-normalized logit
hashes and exact greedy token traces, stopping at EOS or the token bound. Real
loopback HTTP checks use the ordinary handler and Runtime: nonstream completion
matches the direct trace, stream matches text and terminates with DONE, unload
releases host/GPU residency, and disk reload reproduces text/usage. This gate
does not cover chat, cancellation, concurrent requests, graph execution,
long contexts or cross-model swapping.

The combined run has a 120-second deadline, individual HTTP reads time out at
10 seconds, and the outer guard can kill the entire owned group sooner. A cold
kernel build may exhaust this budget; that is a failed run, not qualification.
Per-phase logs/results and an initially-failed aggregate result are retained;
an externally killed run never becomes a pass. Acceptance requires **both**
the aggregate result and outer guard to succeed. Timings only describe bounded
execution; `performance_qualified` is always false.

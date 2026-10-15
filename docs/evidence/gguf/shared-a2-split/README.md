# Shared A2 split-GGUF gate (#218)

Runner revision: a81ea47. Source checkpoint: Qwen/Qwen2.5-0.5B-Instruct-GGUF,
revision 9217f5db79a29953eb74d5343926648285ec7e67,
`qwen2.5-0.5b-instruct-q8_0.gguf`. The retained preparation script uses the
installed upstream GGUF Python writer to create two standard numbered shards.
It does not invoke an upstream inference engine. CPU preparation found exact
packed bytes and metadata across all 291 tensors, with raw prompt IDs [9707].

Before the run, A2 free memory was 4289 MiB and temperature 54 C. Existing GLM
PID 1362245 occupied 10668 MiB on A2 and 22088 MiB on A30, with healthy HTTP
readiness. Explicit busy-service admission permits active requests. No service,
driver, partition, power or fan configuration is changed. Only the A2 UUID is
visible to the owned experiment.

Limits: context128, four output tokens, no graphs, one model resident, four
sequential fresh processes, 1536 MiB PyTorch allocator cap, total120 seconds.
External guard preserves2560 MiB free GPU memory and16 GiB host headroom,
stops at65 C, and checks incumbent health and identity. These are containment
checks, not compute isolation or proof of zero incumbent latency impact.

## First attempt

Both numerical traces passed with identical logits/tokens. The HTTP worker's
initial request hit its10-second response timeout while importing the cold
model; shutdown then reported the still-owned request. The outer run failed
and reaped its owned processes after93 seconds. The guard's final sample had
4289 MiB free and55 C; GLM readiness remained ok and its GPU allocations were
unchanged. Failed reports/log and exact numerical traces are retained here.

The runner now explicitly loads the model under the existing120-second outer
deadline before HTTP generation. The10-second HTTP response timeout, allocator
cap and all resource guards remain unchanged. A loopback regression requires
explicit load before the first generation request.

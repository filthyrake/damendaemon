# Shared CUDA admission

Issue #216 adds `python -m scripts.experiments.guard_shared_cuda` for short,
explicitly owned correctness checks on a CUDA device already serving traffic.
It does not partition the GPU or guarantee incumbent latency. Polling cannot
prevent transient contention, allocation races, or driver-wide failures.

The launcher pins one UUID, clears inherited DLLM configuration, limits CPU
thread defaults, requires a known incumbent PID/start time, healthy `/health`,
and a nonempty `/slots` response with every `is_processing` exactly false.
It requires 4096 MiB free GPU memory initially, preserves 2560 MiB thereafter,
rejects temperatures at 65 C, and preserves 16 GiB available host RAM. A command
must have its own allocator cap and bounded workload. The launcher neither
changes service settings nor supplies hard memory/compute isolation.

Deadline is 1–120 seconds. Each health request has a one-second socket timeout,
and nvidia-smi a two-second timeout; monitoring takes place between probes at
250 ms intervals. Blocking telemetry can delay detection. Failure terminates
only the new process group with SIGKILL and reaps its leader. Commands must not
daemonize or escape that group. There is no automatic retry or background load.

## Live admission evidence, 2026-10-15

On labnode3, the A2 had 4289 MiB free and was 53 C. The existing GLM service
shared A2 and A30, PID 1362245. Its single slot was actively processing a long
request. The harmless child `python3 -c 'print(123)'` was **never launched**:
[a2-busy-admission.json](a2-busy-admission.json) records fail-closed admission.
No CUDA allocation or inference was attempted.

After refusal, `/health` remained `{"status":"ok"}`. NVIDIA telemetry retained
the same incumbent PID with 22088 MiB on A30 and 10668 MiB on A2. These checks
prove admission refusal and continued readiness, not latency preservation or
successful shared inference.

Five focused CPU tests cover malformed/busy service responses, initial and
runtime memory/temperature/identity rejection, deadline, successful cleanup,
and new-work interruption with a separate live process left running.

Real split-GGUF numerical and HTTP qualification remains open under #43. It
requires a separate bounded runner, an allocator budget, exact checkpoint and
shard equivalence, and successful fresh idle admission. Shared-run timings are
not performance qualification. HOMER requires its separate fan/thermal guard.

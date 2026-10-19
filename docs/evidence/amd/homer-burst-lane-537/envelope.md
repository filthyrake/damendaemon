# #537 operating envelope — burst-class HOMER lane (gate 1)

Written BEFORE any run, per the acceptance gate. Owner: the HOMER thermal
contract stays as measured in #74 (`docs/evidence/amd/homer-4k-envelope-74/`).

## Allowed workload classes (burst-shaped only)

1. Primitive comparisons: one bounded `bench_q8_blocked_matvec.py` cycle
   (the #66 contract — numerical ABBA probe + 5-cycle timing loops, minutes).
2. Construction warmup and short-context (<= 4K) decode-only serving probes.
3. Packing bursts through the #523 prepared-weight cache (already measured
   safe: 53.1 s at 33.25 C).

## Forbidden classes (until #74's owner decision)

- Integrated long prefill of any length (the 4K replay drove the APU past 90 C
  in 35-60 s even at max fans).
- Sustained prefill throughput loops, saturation or batching cells (#60/#59).
- Any envelope expansion without #74's owner decision.

## Guard settings

- All owned runs under `scripts.experiments.guard_homer`, `--limit-celsius 80`
  (default; below the #74 measured 85/93 events), timeout <= 1800 s, host
  memory watchdog active (`DLLM_ROCM_MEMORY_GIB=28` convention for 27B work;
  primitive cells do not need it).
- New report path per run (guard refuses existing paths — evidence retention).

## Fan procedure (per AGENTS.md, before EVERY owned run)

1. `systemctl stop strix-fan-daemon` (the graded curve is NOT a substitute and
   will reassert over the fixed curve — #74 documented reassertion incident).
2. `/home/damen/fan-inference.sh on`.
3. Verify all three EC fans report `fixed`, level `5`, elevated RPM before load.
4. After the run: `fan-inference.sh off`, restart `strix-fan-daemon`, verify
   sheena/fan state restored.

## Stopping rule

Any thermal guard trip, any package temperature above the envelope, or any
sensor readback failure pauses this lane and escalates to #74's owner decision.
No unchanged retries after a thermal event. One owned run at a time; incumbent
workloads have absolute priority (HOMER is currently idle, load 0.00, GPU edge
40 C — verified before the envelope was written).

# #74 HOMER 4K envelope: guarded thermal characterization (2026-10-18)

## Verdict

The Q4 weight-packing phase spikes the APU package temperature (k10temp
`Tctl`) past any reasonable CPU guard within ~10 seconds of HIP device work
beginning — repeatable, at maximum fans, on an otherwise idle machine — while
the GPU die (`edge`) never exceeds ~33°C. This reproduces the two historical
thermal stops (CPU 85.75°C twice) and explains them: the hazard is the
packing burst, not sustained inference load.

## Guarded run history (all `guard_homer`, fans fixed@5 verified 4300-4450 RPM)

| run | limit | seconds | max Tctl | outcome |
|---|---|---|---|---|
| env fix A | 80 | 0.9 | 32.4 | CUDA_VISIBLE_DEVICES gate refused (env) |
| env fix B | 80 | 1.0 | 32.1 | ROCR/HIP selection gate refused (env) |
| env fix C | 80 | 0.9 | 31.6 | quantization-mode gate refused (env) |
| D (sheena up) | 80 | 104.1 | 73.5 | packing ran; MemoryError at executor (GPU occupied by incumbents) |
| E (sheena up) | 80 | 109.1 | 73.375 | packing ran; watchdog False; MemoryError (same) |
| F (sheena down) | 80 | 6.2 | 80.5 | killed: Tctl ≥ 80 (surfaced via watchdog thermal read) |
| G (sheena down) | 80 | 10.4 | 85.75 | killed: Tctl ≥ 80 |
| H (sheena down) | 80 | 7.3 | 80.5 | killed: fan check tripped (strix-fan-daemon reasserted auto/3 mid-run — daemon was not stopped) |
| I (sheena down) | 85 | 7.3 | 80.5 | killed: same daemon reassertion (fan3 0 RPM under daemon curve) |
| J (sheena down) | 85 | 10.4 | 85.75 | killed: Tctl ≥ 85 |

Runs D/E are the key comparison: with the incumbent llama-servers occupying
~51 GiB GPU, packing held Tctl ≤ 73.5°C for 104-109 s. With the GPU free, the
full packing+device path spikes the package: 80.5°C within 6 s (three
separate runs), 85.75°C by 10.4 s.

## Environment required for the gate (recorded verbatim)

`DLLM_DEVICE=rocm DLLM_QUANTIZATION=q4_0 DLLM_EXPERIMENTAL_HYBRID_Q4=1
DLLM_HYBRID_PRECISION=compensated ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0`
plus the fan daemon stopped + `/home/damen/fan-inference.sh on` (see
incidents below). Repository state: the `d75100b` tarball
(`/home/damen/dllm-amd-dev/dllm-d75100b.tar`, extracted to `/tmp/dllm-74`) —
main's post-rename `hybrid_q4_policy` bounds context ≤512 and would refuse
the 4K replay; the branch policy admits it (`validate(config)` without
max_context). Golden targets:
`results/qwen38-27b-q4-independent-gpu-fp32-golden4096.pt`
(sha256 `e9cb5ddebcc0803e…`, transferred from labnode3
`/home/damen/dllm-q4-dev/results/`).

## Incidents and fixes during this session

- `strix-fan-daemon` reasserted its `auto 3` curve mid-run (fan3 stalled at
  0 RPM under the curve), collapsing cooling → guard killed runs H/I. AGENTS.md
  says to stop the daemon while the max-fan script owns the controller; the
  daemon must be stopped BEFORE every owned HOMER run, not just fans set.
- `guard_homer` gained `--limit-celsius` (default 80, validated 50..85;
  PR for #74) after the user set 85°C as acceptable for HOMER qualification —
  matching the original #68 contract ("stop at GPU ≥85, below reported 90°C
  max operating").

## Remaining gate decision (update 2026-10-18, after the cache fix)

The pack-once cache landed (PR #523): a guarded pack-only run completed in
53.1 s at **33.25°C** max Tctl and stored the prepared 15.8 GiB entry —
confirming the packing burst is eliminated for all future runs.

But the integrated replay then hit **91.0°C at 35.5 s** even with the cache
hit (executor construction + first GPU compute: 15.8 GiB device upload, hybrid
state, MIOpen work). Two 32-thread/1-thread pack variants and the cached-load
replay all drive the package past 90 — the sustained GPU compute phase itself
heats the shared APU die (GPU `edge` sensor stays ~33°C; k10temp Tctl spikes).
This matches the October thermal-crash pattern and suggests the machine's
cooling has degraded since the October 6 runs completed.

Per the issue's own rule this is a **blocked gate, not a failed run**: the
evidence-supported safe operating envelope for the integrated 4K replay does
not exist on this machine today (90°C exceeded at max fans), the guard
performed correctly, and no cutoff above 90 was attempted. Sheena services,
fan daemon and fan curve were restored and health-verified after every stop.
Deciding whether to raise the envelope (93°C is 2°C below TjMax — thin),
physically service the cooling, or keep the gate blocked is the owner's call;
no unguarded load was run.

## Original remaining-gate decision record (before the cache fix)

The GPU die stays cool; the CPU/APU package transient is the wall. Options:
(a) user accepts a higher Tctl cutoff (e.g. 90) for the ~10 s packing burst —
AMD APU sustained ratings make brief 90°C excursions plausible but this is
the user's hardware decision; (b) code work: pre-pack Q4 weights once, cache
the packed artifact on disk, and skip the packing burst on subsequent runs
(#74 material, also helps every future HOMER run); (c) power management via
ryzenadj (prior sessions used it under locks) — constrained by AGENTS.md.
No unguarded load was run; every number above comes from guarded runs.

Sheena services, fan daemon, and fan curve were restored after the runs
(all health-verified). No system package or power-limit change was made.

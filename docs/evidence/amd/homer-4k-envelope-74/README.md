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

## Final round (2026-10-18, after the owner cleaned the fans and said push on)

The compressed-air cleaning did NOT move the thermal equilibrium. With the
packing burst eliminated (cache hit), the sustained GPU compute phase still
drove Tctl past the limit in every configuration tried:

| run | power caps (STAPM/fast/slow/APU) | limit | seconds | max Tctl |
|---|---|---|---|---|
| cache-hit replay | 120/140/120/70 (as found) | 90 | 35.5 | 91.0 |
| cache-hit replay | 45/45/45/70 | 90 | 32.2 | 90.0 |
| cache-hit replay | 45/45/45/45 | 90 | ~55 | 90.0 (sampler curve) |
| cache-hit replay | 45/45/45/45 | 93 | 43.3 | 94.0 |

The Tctl curve is a smooth sustained ramp (35→90 over 28-60 s depending on
power caps), not a burst. Crucially, re-reading the October evidence shows
**the October completing runs never recorded temperatures** — the two October
runs WITH thermal data failed at 85.75°C@16.7 s and 80°C@124.8 s, exactly
matching today. This machine has plausibly always ridden ~85-94°C on this
workload; the completing runs predate measurement.

Also discovered and recorded: the box was configured at STAPM 120 W / PPT fast
140 W (vs the 45 W the October optimization sessions used). Power caps change
the ramp speed but not the equilibrium — even 45/45/45/45 reaches 90-94°C.
The likely cause is the die's thermal path (compound/mounting), not fans —
compressed air made no difference. TjMax is 95°C; the SMU emergency shutdown
sits beyond it, which is how the October hard crash happened.

**Verdict: #74's integrated 4K gate stays blocked.** No configuration tried
(today or in October) holds the workload under 90°C. The gate needs either
physical thermal service (repaste/mounting — beyond compressed air) or an
explicit owner acceptance of ~94°C operation with the emergency-shutdown risk
that caused the October crash. Everything needed to retry after service is in
this README and the skill: the guard command, env contract, cached weights
(stored at /home/damen/weight-cache — the packing burst is gone forever), and
the 45W ryzenadj recipe.

Original remaining-gate decision record (before these rounds):

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

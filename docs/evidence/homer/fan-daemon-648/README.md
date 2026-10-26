# strix-fan-daemon v2 — shadow validation cell (#648)

Date: 2026-10-26. Host: HOMER (Radeon 8060S / gfx1151, kernel 7.2.12-1.el9.elrepo).
Deliverable: `scripts/strix-fan-daemon-v2.sh` — v1-equivalent Tctl PID curve
plus SMC gpu-metrics-blob inputs (dual-die temps, socket-power feedforward),
**shadow-qualified only**: the cell ran v2 in logging mode with no real fan
writes; production v1 (`strix-fan-daemon.service`) stayed live and untouched
throughout (verified `active`, holding AUTO at idle, after the cell).

## What v2 adds over v1

1. **Dual-die control temperature**: `control = max(Tctl, hottest gpu_metrics
   die temp)`. The blob's two temps lead Tctl under GPU load (cell: gfx hit
   52 C at 11:46:42 while Tctl read 51; by load end gfx 58 = Tctl 58).
2. **Power feedforward**: while socket power is sustained ≥ 40 W, fans are
   floored at level 3 ahead of the thermal ramp. Engagement is 3-of-last-5
   polls ≥ 40 W; release is 3 consecutive polls < 40 W.
3. **Degrade-to-v1**: any blob format surprise (unknown revision, truncated
   blob, implausible values) drops the blob inputs silently and leaves the
   v1 Tctl-only curve in sole control.

## Blob layout (pinned, with the corrections that cost two cell runs)

`/sys/class/drm/card0/device/gpu_metrics` — format_revision 3,
structure_size 264 bytes, matches the amdgpu `gpu_metrics_v3_0` struct:

| field | offset | type |
|---|---|---|
| structure_size | 0 | u16 |
| format_revision | 2 | u16 |
| die temp A | 4 | u16 (0.01 C) |
| die temp B | 6 | u16 (0.01 C) |
| system_clock_counter | 104 | u64 (ns) |
| average_socket_power | 112 | **u32** LE (mW) |
| average_apu_power | 120 | u32 LE (mW) |
| average_gfx_power | 124 | **u32** LE (mW) |
| average_gfxclk_frequency | 174 | u16 (MHz) |
| average_uclk_frequency | 186 | u16 (MHz) |

Cross-validated 2026-10-26 against umr (`--power`) at idle and under load,
and against a full `od -v` dump (`blobdump.txt`): socket 76,977 mW (blob) vs
~80 W (umr) under sustained GEMM; gfx power 23.3 W; uclk 1000 MHz = 8000
MT/s LPDDR5X; gfxclk 2900 MHz under load.

### Bugs found and fixed (all three preserved as findings)

1. **od `*` dedup compression**: `od -An -tu2 -N 128` emits a `*` line for
   repeated rows (57 words for a 64-word region), breaking any whole-stream
   word indexing — the first parser indexed awk words and silently read the
   wrong fields. Fix: every field is read at a pinned byte offset
   (`od -j <byte> -N <len>`), one read per field, immune to compression.
2. **u16 vs u32 socket power**: the first byte-offset parser read
   `average_socket_power` as u16. It is a **u32**; the u16 read captured
   only the low 16 bits, so ≥ 40 W wrapped to < 16 W
   (76,977 mod 65,536 = 11,441 "mW" while umr read ~77 W) — at idle
   (< 65,536 mW) the low half alone looked correct, which is why the parser
   test passed but the feedforward never engaged under load. Fix: `-tu4`
   4-byte reads, plus a plausibility gate (< 500 W, temps < 150 C →
   otherwise degrade to v1).
3. **bash window truncation**: `${fw_window: -FW_WINDOW}` on a string
   *shorter* than 5 chars returns EMPTY in bash, so the rolling window was
   zeroed every poll and `fw_high` never reached 3. Fix: append first, then
   drop the oldest char only when the window exceeds `FW_WINDOW`.

### Feedforward window design (from measurement, not intuition)

The blob's `average_socket_power` is a ~1 s SMC moving average. Real bursty
GPU load (GEMM batches with gaps) oscillates the average across the 40 W
line poll-by-poll — under a bursty probe (`fw-burst-probe.py`), 3 *consecutive*
≥ 40 W polls never occurred (best measured run = 1). Engagement therefore
uses 3-of-last-5 (robust to burst gaps, still rejects one-off spikes);
release keeps 3 consecutive below-threshold polls.

## Final cell result (v2-shadow.log, umr.log, load.log, cell.log)

Cell: 30 s idle observe → 100 s sustained GEMM (512×5120×17408 f32,
415 iterations) → 90 s cool-down. umr sampled at 2 Hz as the independent
observer; v2 ran in shadow mode (decision log only).

- 11:46:04 start, blob online (rev 3), AUTO at Tctl 39 C.
- 11:46:42 **feedforward ENGAGED: socket 78,336 mW (3-of-5) → floor level 3**
  (~8 s into the load) — Tctl 51, gfx 52.
- Load window: heartbeats show `fw=1 lvl=3` throughout, Tctl/gfx 54→58 C.
- 11:48:23 **feedforward RELEASED: socket 17,983 mW** → AUTO, ~9 s after the
  load ended (3 release polls).
- Cool-down: Tctl 42 C by 11:49, `fw=0 lvl=0`.
- Idle phases never engaged (socket 15–18 W << 40 W threshold).

umr corroborates the same run: socket median 80.25 W during the load window,
188/418 samples ≥ 40 W, sustained — versus ~16.5 W idle.

## Qualification matrix

| check | result |
|---|---|
| parser vs umr, idle (4 samples, 2 s apart) | agree within ~500 mW / temp deltas ≤ 0.5 C (`daemon-parser-test.sh`) |
| parser vs umr, under GEMM load | 76,977 vs ~80,000 mW (independent ~1 s SMC windows) |
| feedforward engage under sustained load | engaged ≤ 8 s, held through load |
| feedforward idle rejection | never engaged at 15–18 W idle |
| feedforward release | ~9 s after load end |
| degrade: missing blob / short blob | rc=2 → v1 behavior, logged once |
| degrade: unknown format revision | rc=2 (same path) |
| production v1 during cell | `active`, AUTO at idle, log clean |

## Files

- `v2-shadow.log` — final passing cell, full v2 decision log
- `umr.log` — 2 Hz independent observer (temp A/B, socket, gfx power, mW)
- `d648-cell.sh`, `load-live.log` (live cell), `d648-load.py` — load generator + cell harnesses
- `blobdump.txt` — full `od -v` blob dump under load (layout derivation)
- `daemon-parser-test.sh` — parser-vs-umr + degrade-path checks
- `fw-burst-probe.py` — bursty-load probe that motivated the 3-of-5 window
- `cell-live.log`, `v2-live.log`, `load-live.log`, `d648-live-cell.sh` — the
  live real-fan cell (below)
- `unit-dropin-v2.conf` — the production `ExecStart` drop-in (verbatim)

Not retained: the two failing shadow logs were overwritten by relaunches;
their observed failure signatures (od `*` word-misalignment; `fw=0` at
~97.8 W umr peak with blob socket stuck ~11–13 W; 57-word od output) are
recorded above and in issue #648.

## Live fan cell + production switch (2026-10-26, same day)

The real-fan cell (`cell-live.log`, `v2-live.log`, `d648-live-cell.sh`,
`unit-dropin-v2.conf`) ran v2 with **real EC writes**, production v1
suspended for its ~5-minute duration and restarted by the teardown trap:

- Baseline (v1): fans auto, 1051/1022/0 RPM at Tctl 39.9 °C.
- v2 takeover: clean start, AUTO at Tctl 39.
- **ENGAGED 12 s into the 100 s GEMM load**: socket 77,576 mW (3-of-5) →
  all three fans `fixed` level 3; RPM 1051/1022/0 → **3180/3200/1190**
  (fan 3 spun up from standstill); Tctl held ≤ 53 °C through the load.
- **RELEASED** ~28 s after load end: 15,604 mW, 3 consecutive lows → AUTO;
  RPM settling 2003/2213/610 at the cool-down snapshot, 1047/1019/0 at idle.
- Teardown `PASS=1` (engage + release + auto all observed); v1 restarted
  automatically; no cell processes left behind.

**Production switch (user-approved)**: v2 installed at
`/usr/local/sbin/strix-fan-daemon-v2.sh` (v1 left in place) and the
`strix-fan-daemon.service` `ExecStart` redirected via a systemd drop-in
(`unit-dropin-v2.conf`, `ExecStart=` reset + new path — the unit's own
Restart=always fail-safe is unchanged). Verified post-swap: service
`active` on the v2 script (journal shows the v2 banner, blob online, AUTO
at Tctl 35 °C, clean heartbeats), fans auto at idle RPMs.

**Rollback** (one command, fans hold on daemon death per the unit's
fail-safe): `sudo rm /etc/systemd/system/strix-fan-daemon.service.d/v2.conf
&& sudo systemctl daemon-reload && sudo systemctl restart strix-fan-daemon`.

#!/bin/bash
# strix-fan-daemon v2 — dual-die + power feedforward (#648).
#
# Changes vs v1 (strix-fan-daemon.sh):
#   - Control variable is max(Tctl, hottest gpu_metrics die temp) instead of
#     Tctl alone: the gfx die leads during GPU inference; Tctl lags seconds
#     behind.
#   - Power feedforward: sustained socket power floors the fans at FW_LEVEL,
#     because power steps the instant load lands while temperature only
#     moves after the package is already heating. Engagement is FW_NEED-of-
#     FW_WINDOW polls >= FW_WATTS (NOT consecutive-only — see the measured
#     note at the tunables below); release is FW_RELEASE consecutive polls
#     below the threshold.
#   - gfx die temps + socket/gfx power come from the amdgpu gpu_metrics
#     blob (/sys/class/drm/card0/device/gpu_metrics, root-readable — the
#     daemon runs as root for the EC writes regardless). Layout pinned
#     2026-10-26 against the gpu_metrics_v3_0 struct and cross-validated
#     against umr: dual-die temps u16 @ bytes 4/6 (0.01 C), socket power
#     u32 LE @ 112 (mW), gfx power u32 LE @ 124 (mW). SEE read_gpu_metrics:
#     socket power is a u32 — a u16 read wraps >=40 W to <16 W. Any unknown
#     format_revision, short/absent blob, or implausible values degrade the
#     daemon to the v1 Tctl-only behavior (logged once per state change).
#   - Polls every 2s (blob reads are cheap pinned-offset od calls).
#
# Unchanged from v1 by design: the graded curve anchors (75/79/82/85 ->
# 2/3/4/5), 3C hysteresis, never touching apu/power_mode or ryzenadj, and
# the fail-safe (daemon death holds fans; systemd restarts).
#
# Validation: docs/evidence/homer/fan-daemon-648/ (shadow cell, umr
# cross-checks, degrade-path checks, and the three parser bugs found).

set -u

EC="/sys/class/ec_su_axb35"
GPU_METRICS="/sys/class/drm/card0/device/gpu_metrics"

# ---- tunables ---------------------------------------------------------------
INTERVAL=2            # seconds between polls
HYST=3                # downward deadband in C
HEARTBEAT=30          # log every N polls even when nothing changes
CURVE=(75:2 79:3 82:4 85:5)   # Damen's anchors — unchanged
# Power feedforward: floor the fans at level 3 while socket power stays at
# or above 40W for 3 consecutive polls; release when it drops below for 3.
# 40W is above the measured idle envelope (~16-33W) and below the #74-class
# GPU decode load (~40W+); calibrate once against the validation cell.
# NOTE (measured 2026-10-26): the blob's average_socket_power is a ~1s SMC
# average, and real bursty GPU load (GEMM batches with gaps) oscillates the
# average across the 40W line poll-by-poll — 3 CONSECUTIVE >=40W polls never
# happen under bursty load (measured: best run = 1). Engagement therefore
# uses a 3-of-last-5 window (robust to bursty gaps, still rejects single
# one-off spikes); release keeps 3 consecutive below-threshold polls.
FW_WATTS=40000        # mW threshold
FW_WINDOW=5           # polls considered
FW_NEED=3             # polls >= threshold within the window to engage
FW_LEVEL=3            # floor level while sustained
FW_RELEASE=3          # consecutive polls below threshold before release
# ------------------------------------------------------------------------------

log() { echo "$(date '+%F %T') $*"; }

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "strix-fan-daemon must run as root (it writes $EC)." >&2
  exit 1
fi

find_tctl() {
  local h lab
  for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/name" ] || continue
    [ "$(cat "$h/name")" = "k10temp" ] || continue
    for lab in "$h"/temp*_label; do
      [ -r "$lab" ] || continue
      if [ "$(cat "$lab")" = "Tctl" ]; then echo "${lab%_label}_input"; return 0; fi
    done
    [ -r "$h/temp1_input" ] && { echo "$h/temp1_input"; return 0; }
  done
  return 1
}

# Read the gpu_metrics blob; echoes "t1_mC t2_mC socket_mW gfx_mW" or fails.
# Layout pinned 2026-10-26 by full-blob dump (od -v, no dedup compression)
# cross-validated against umr and the amdgpu gpu_metrics_v3_0 struct
# (format_revision 3, structure_size 264): dual-die temps u16 @ bytes 4/6
# (0.01 C, both track GPU load; hottest is taken), average_socket_power
# u32 LE @ 112 (mW), average_gfx_power u32 LE @ 124 (mW). CORRECTED from the
# first draft, which read socket power as u16 — that captured only the low
# 16 bits of the u32, so >=40 W wrapped to <16 W and the feedforward never
# engaged (caught by the validation cell: blob showed 11441 mW while umr
# showed ~77 W; 76977 mod 65536 = 11441). Unknown format revision or a
# short blob is a hard degrade (offsets may move). Pure od+awk; each read
# is at a pinned byte offset, immune to od's "*" dedup-line compression.
read_gpu_metrics() {
  local size rev
  # Header: structure_size u16 @0, format_revision u16 @2.
  size=$(od -An -tu2 -j 0 -N 2 "$GPU_METRICS" 2>/dev/null | awk '{print $1}')
  rev=$(od -An -tu2 -j 2 -N 2 "$GPU_METRICS" 2>/dev/null | awk '{print $1}')
  if [ "$rev" != "3" ] || [ -z "$size" ] || [ "$size" -lt 128 ]; then
    return 2   # unknown format revision or truncated — degrade to v1
  fi
  local t1 t2 socket gfxp
  t1=$(od -An -tu2 -j 4 -N 2 "$GPU_METRICS" 2>/dev/null | awk '{print $1}')
  t2=$(od -An -tu2 -j 6 -N 2 "$GPU_METRICS" 2>/dev/null | awk '{print $1}')
  socket=$(od -An -tu4 -j 112 -N 4 "$GPU_METRICS" 2>/dev/null | awk '{print $1}')
  gfxp=$(od -An -tu4 -j 124 -N 4 "$GPU_METRICS" 2>/dev/null | awk '{print $1}')
  [ -n "$t1" ] && [ -n "$t2" ] && [ -n "$socket" ] && [ -n "$gfxp" ] || return 1
  # Sanity: socket power must be a plausible u32 mW value (< 500 W) and a
  # temp must be plausible (< 150 C) — guards against a layout change on a
  # format-compatible revision.
  if [ "$socket" -ge 500000 ] || [ "$t1" -ge 15000 ] || [ "$t2" -ge 15000 ]; then
    return 2
  fi
  echo "$t1 $t2 $socket $gfxp"
}

desired_level() {
  local t=$1 lvl=0 entry min level
  for entry in "${CURVE[@]}"; do
    min=${entry%:*}; level=${entry#*:}
    [ "$t" -ge "$min" ] && lvl=$level
  done
  echo "$lvl"
}

set_level() { # 0 = auto
  local lvl=$1 fan
  for fan in "$EC"/fan*; do
    [ -d "$fan" ] || continue
    echo fixed > "$fan/mode" 2>/dev/null
    echo "$lvl" > "$fan/level" 2>/dev/null
  done
}

TCTL=$(find_tctl) || { log "FATAL: no k10temp Tctl found"; exit 1; }

blob_ok=-1          # -1 unknown, 1 ok, 0 degraded (log transitions once)
fw_high=0           # polls >= threshold within the current window
fw_window=""        # recent poll booleans (1/0), FW_WINDOW entries
fw_low=0            # consecutive polls below the threshold
fw_on=0             # feedforward floor currently engaged
cur=-1              # current applied level (-1 unknown)
poll=0

log "strix-fan-daemon v2 starting: Tctl=$TCTL curve=(${CURVE[*]}) fw=${FW_WATTS}mW ${FW_NEED}-of-${FW_WINDOW}->$FW_LEVEL"

while true; do
  tctl=$(( $(cat "$TCTL" 2>/dev/null || echo 0) / 1000 ))

  control=$tctl
  gfx_mC=""; socket_mW=""
  if gm=$(read_gpu_metrics); then
    if [ "$blob_ok" -ne 1 ]; then log "gpu_metrics blob online (rev 3)"; blob_ok=1; fi
    # gm = "t1_mC t2_mC socket_mW gfx_mW"; control uses the HOTTEST die temp.
    gfx_mC=$(echo "$gm" | awk '{print ($1 > $2) ? $1 : $2}')
    socket_mW=$(echo "$gm" | awk '{print $3}')
    gfx=$(( gfx_mC / 100 ))
    [ "$gfx" -gt "$control" ] && control=$gfx
  else
    rc=$?
    if [ "$blob_ok" -ne 0 ]; then
      log "gpu_metrics unavailable (rc=$rc) — degraded to Tctl-only (v1 behavior)"
      blob_ok=0
    fi
  fi

  lvl=$(desired_level "$control")

  # Power feedforward (only when the blob works — power is blob-only data).
  # NOTE: the window is a bit-string; we append then drop the OLDEST char
  # only when longer than FW_WINDOW (a negative-offset substring on a
  # shorter string returns EMPTY in bash — that bug zeroed the window every
  # poll and was caught by the first validation cell).
  if [ -n "$socket_mW" ]; then
    if [ "$socket_mW" -ge "$FW_WATTS" ]; then
      fw_window="${fw_window}1"
      fw_low=0
    else
      fw_window="${fw_window}0"
      fw_low=$(( fw_low + 1 ))
    fi
    if [ "${#fw_window}" -gt "$FW_WINDOW" ]; then
      fw_window=${fw_window:1}
    fi
    fw_high=$(echo "$fw_window" | awk '{n=gsub(/1/,""); print n}')
    if [ "$fw_on" -eq 0 ] && [ "$fw_high" -ge "$FW_NEED" ]; then
      fw_on=1
      log "feedforward ENGAGED: socket ${socket_mW}mW (${fw_high}-of-${FW_WINDOW}) — floor level $FW_LEVEL"
    elif [ "$fw_on" -eq 1 ] && [ "$fw_low" -ge "$FW_RELEASE" ]; then
      fw_on=0
      log "feedforward RELEASED: socket ${socket_mW}mW below threshold"
    fi
  else
    fw_on=0; fw_window=""
  fi
  [ "$fw_on" -eq 1 ] && [ "$lvl" -lt "$FW_LEVEL" ] && lvl=$FW_LEVEL

  # Hysteresis: only drop a level if control is HYST below the level's entry.
  if [ "$cur" -gt "$lvl" ]; then
    drop_ok=0
    for entry in "${CURVE[@]}"; do
      min=${entry%:*}; level=${entry#*:}
      if [ "$level" -eq "$lvl" ] && [ "$control" -le $(( min - HYST )) ]; then
        drop_ok=1
      fi
    done
    [ "$lvl" -eq 0 ] && [ "$control" -le $(( 75 - HYST )) ] && drop_ok=1
    [ "$drop_ok" -eq 0 ] && lvl=$cur
  fi

  if [ "$lvl" -ne "$cur" ]; then
    if [ "$lvl" -eq 0 ]; then
      for fan in "$EC"/fan*; do [ -d "$fan" ] && echo auto > "$fan/mode" 2>/dev/null; done
      log "Tctl=$tctl gfx=$([ -n "$gfx_mC" ] && echo $(( gfx_mC / 100 )) || echo "-") control=$control -> AUTO"
    else
      set_level "$lvl"
      log "Tctl=$tctl gfx=$([ -n "$gfx_mC" ] && echo $(( gfx_mC / 100 )) || echo "-") control=$control socket=$([ -n "$socket_mW" ] && echo "$socket_mW" || echo "-") -> level $lvl"
    fi
    cur=$lvl
  fi

  poll=$(( poll + 1 ))
  if [ $(( poll % HEARTBEAT )) -eq 0 ]; then
    log "heartbeat poll=$poll Tctl=$tctl gfx=$([ -n "$gfx_mC" ] && echo $(( gfx_mC / 100 )) || echo "-") control=$control lvl=$cur fw=$fw_on"
  fi

  sleep "$INTERVAL"
done

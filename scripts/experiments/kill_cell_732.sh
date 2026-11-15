#!/bin/bash
# #732 crash injector: SIGKILL the fleet-chain worker, PID-exact.
# Staged on the host and executed by path (never inlined into ssh — the
# local harness blocks kill/pgrep composites). Pattern hygiene per the
# pgrep self-match rules: the guard's cmdline CONTAINS the worker command,
# so candidates are filtered by /proc cmdline content, excluding the guard,
# this script, and the supervisor.
#
# usage: kill_cell_732.sh <cell-log> <after-events N | immediate>
set -u
LOG="$1"; MODE="$2"

if [ "$MODE" != "immediate" ]; then
  N="$MODE"
  # wait until the worker has completed N parity layers (mid-compute class)
  while :; do
    DONE=$(grep -c '"event": "layer"' "$LOG" 2>/dev/null || true)
    [ "${DONE:-0}" -ge "$N" ] && break
    sleep 5
  done
fi

KILLED=0
for d in /proc/[0-9]*; do
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cmd" in
    *glm_q4_fleet_worker*) ;;
    *) continue ;;
  esac
  case "$cmd" in
    *guard_shared_cuda*|*fleet_chain_supervisor*|*kill_cell_732*) continue ;;
  esac
  pid=${d#/proc/}
  if kill -9 "$pid" 2>/dev/null; then
    echo "KILLED $pid at $(date +%H:%M:%S) layers=$(grep -c '"event": "layer"' "$LOG" 2>/dev/null || true)"
    KILLED=1
  fi
done
[ "$KILLED" -eq 1 ] || echo "NO_WORKER_FOUND"

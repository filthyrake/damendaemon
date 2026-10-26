#!/bin/bash
# #648 validation cell: v1 daemon (live, controlling fans) vs v2 daemon
# (shadow mode — full decision logic, fan writes suppressed) under the same
# bounded GPU load, with the umr 2 Hz sampler as the independent observer.
#
# Phases:
#   idle-observe (30s) -> load (d645 gate-1 GEMM runner, ~5min) -> cool-down
#     -> report JSON
# Measures per daemon: time-to-first-level-change from load onset,
# level decisions over time, control variables (Tctl / gfx / socket W).
# The v2 instance logs its decisions to /tmp/d648/v2-shadow.log.
set -u
D=/tmp/d648
mkdir -p $D

# --- umr sampler (independent observer, 2 Hz) ---
UMR=/home/damen/umr/build/src/app/umr
echo "t	gfx_temp	soc_temp	socket_power	gfx_power" > $D/umr.log
( START=$(date +%s.%N)
  while [ ! -f $D/STOP ]; do
    T=$(echo "$(date +%s.%N) - $START" | bc)
    LINE=$(sudo -n $UMR --gpu-metrics 2>/dev/null | awk '
      /temperature_gfx/ {gfx=$3+0} /temperature_soc/ {soc=$3+0}
      /average_socket_power/ {sock=$3+0} /average_gfx_power/ {g=$3+0}
      END {printf "%.1f\t%.1f\t%d\t%d", gfx/100, soc/100, sock, g}')
    echo -e "$T	$LINE" >> $D/umr.log
    sleep 0.5
  done ) &
SAMPLER=$!
echo "sampler $SAMPLER"

# --- v2 daemon in shadow mode: full logic, no fan writes ---
# Shadow = set EC to a bogus path so writes fail silently while all
# decisions still log; the REAL blob path stays for reads.
sed 's|^EC=.*|EC="/tmp/d648/shadow-ec"|; s|^INTERVAL=.*|INTERVAL=2|' /tmp/daemon-test.sh > $D/v2-shadow.sh
mkdir -p $D/shadow-ec   # exists but empty -> set_level writes fail, auto-release no-ops
chmod +x $D/v2-shadow.sh
setsid nohup sudo -n $D/v2-shadow.sh > $D/v2-shadow.log 2>&1 < /dev/null &
echo "v2 shadow started"
sleep 3
tail -2 $D/v2-shadow.log

echo "=== idle observe 30s ==="
sleep 30

echo "=== load phase: sustained GEMM 100s ==="
cd /tmp/d648/src
PYTHONPATH=/tmp/d648/src /home/damen/ai_benchmarking_project/.venv/bin/python /tmp/d648-load.py 100 > $D/load.log 2>&1
echo "load done rc=$?"

echo "=== cool-down 90s ==="
sleep 90

touch $D/STOP
sleep 1
kill $SAMPLER 2>/dev/null
sudo -n pkill -f v2-shadow.sh 2>/dev/null
echo "cell complete"

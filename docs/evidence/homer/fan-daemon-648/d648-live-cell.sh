#!/bin/bash
# #648 LIVE fan cell: v2 with REAL EC writes (production v1 suspended for the
# duration). Phases: baseline -> v2 real-mode takeover -> sustained GEMM load
# (expect ENGAGE -> fixed level 3 + RPM jump) -> cool-down (expect RELEASE ->
# auto + RPM settle) -> teardown. Any failure path restarts v1.
set -u
D=/tmp/d648-live
V2=$D/strix-fan-daemon-v2.sh
EC=/sys/class/ec_su_axb35
PY=/home/damen/ai_benchmarking_project/.venv/bin/python

snap() {
  for f in $EC/fan*/; do
    printf "fan=%s mode=%s level=%s rpm=%s " "${f%/}" "$(cat $f/mode)" "$(cat $f/level)" "$(cat $f/rpm)"
  done
  printf "Tctl=%s\n" "$(cat /sys/class/hwmon/hwmon3/temp1_input)"
}

restore_v1() {
  sudo -n pkill -f "$V2" 2>/dev/null
  sleep 1
  sudo -n systemctl start strix-fan-daemon
  echo "$(date '+%F %T') RESTORED v1: $(sudo -n systemctl is-active strix-fan-daemon)"
}

echo "=== phase 0: baseline (v1 active) ==="
sudo -n systemctl is-active strix-fan-daemon
snap

echo "=== phase 1: v1 stop, v2 real-mode takeover ==="
sudo -n systemctl stop strix-fan-daemon
sudo -n setsid nohup bash "$V2" > $D/v2-live.log 2>&1 < /dev/null &
sleep 5
snap
head -3 $D/v2-live.log

echo "=== phase 2: sustained GEMM 100s (expect ENGAGE level 3 + RPM jump) ==="
( sleep 20; echo "t+20s: $(snap)"; sleep 20; echo "t+40s: $(snap)"; sleep 20; echo "t+60s: $(snap)" ) &
$PY $D/d648-load.py 100 > $D/load.log 2>&1
echo "load rc=$?"

echo "=== phase 3: cool-down 60s (expect RELEASE -> auto) ==="
sleep 60
snap
tail -6 $D/v2-live.log

echo "=== phase 4: teardown ==="
if grep -q "ENGAGED" $D/v2-live.log && grep -q "RELEASED" $D/v2-live.log && grep -q "AUTO" $D/v2-live.log; then
  echo "LIVE CELL PASS: engage+release+auto all observed"
  PASS=1
else
  echo "LIVE CELL INCOMPLETE — check $D/v2-live.log"
  PASS=0
fi
sudo -n pkill -f "$V2"
sleep 1
snap
# Leave the box under daemon control either way (caller decides v1 vs v2 unit).
sudo -n systemctl start strix-fan-daemon
echo "v1 restarted: $(sudo -n systemctl is-active strix-fan-daemon)"
echo "PASS=$PASS"

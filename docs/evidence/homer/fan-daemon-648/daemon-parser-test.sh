#!/bin/bash
# Dry-run harness for the v2 daemon's read_gpu_metrics: sources the ACTUAL
# daemon script with the main loop suppressed, then compares its parser
# against umr on live samples. Cannot drift from the shipped parser.
set -u
DAEMON=/tmp/daemon-test.sh

# Source only the functions: execute up to the main-loop marker by
# extracting the script before "TCTL=$(find_tctl)".
eval "$(sed -n '1,/^TCTL=\$(find_tctl)/p' "$DAEMON" | sed '$d')"

GPU_METRICS="/sys/class/drm/card0/device/gpu_metrics"
echo "== parser (sourced from daemon) vs umr, 4 samples 2s apart =="
for i in 1 2 3 4; do
  if gm=$(read_gpu_metrics); then
    echo "blob : $gm"
  else
    echo "blob : FAIL rc=$?"
  fi
  sudo -n /home/damen/umr/build/src/app/umr --gpu-metrics 2>/dev/null | awk '
    /temperature_gfx/ {g=$3} /temperature_soc/ {s=$3}
    /average_socket_power/ {sp=$3} /average_gfx_power/ {gp=$3}
    END {printf "umr  : %d %d %d %d\n", g, s, sp, gp}'
  sleep 2
done

echo "== degrade paths =="
GPU_METRICS="/nonexistent/gpu_metrics"
read_gpu_metrics && echo "UNEXPECTED PASS" || echo "missing-blob degrade ok rc=$?"
printf 'xxxx' > /tmp/short-blob 2>/dev/null || sudo -n sh -c 'printf "xxxx" > /tmp/short-blob'
GPU_METRICS=/tmp/short-blob
read_gpu_metrics && echo "UNEXPECTED PASS" || echo "short-blob degrade ok rc=$?"
rm -f /tmp/short-blob 2>/dev/null || sudo -n rm -f /tmp/short-blob

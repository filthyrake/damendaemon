#!/usr/bin/env bash
# MI300X evidence collector — Phase A (read-only system/stack inventory)
# plus optional Phase B/C (capability probes / microbenchmarks) if a
# torch-bearing python is found. SAFE BY DEFAULT: phase A never allocates
# GPU memory, changes no settings, writes only into its own evidence dir.
#
# Usage:
#   ./collect.sh                 # read-only inventory + capability probes
#   ./collect.sh --bench         # + time-boxed microbenchmarks (one GPU)
#   ./collect.sh --bench --gpu 3 # choose which GPU index benches use
#
# Env overrides: MI300X_PYTHON=/path/to/python (torch env), OUT_DIR=...
set -u
BENCH=0; GPU=0; EXTRA_PY="${MI300X_PYTHON:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH=1 ;;
    --gpu) shift; GPU="${1:-0}" ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

TS=$(date +%Y%m%d-%H%M%S)
HOST=$(hostname -s 2>/dev/null || echo host)
BASE="${OUT_DIR:-mi300x-evidence-$HOST-$TS}"
mkdir -p "$BASE/logs" "$BASE/probes"
echo "evidence dir: $BASE"

# --- helper: run one command with timeout, always record something ---
run() { # run <name> <cmd...>
  local name="$1"; shift
  timeout 120 "$@" >"$BASE/logs/$name.txt" 2>&1 \
    || echo "__EXIT_$?__ (missing/stalled/timeout: $(printf '%q ' "$@"))" >>"$BASE/logs/$name.txt"
  echo "  [.] $name"
}

echo "phase A: system/stack inventory (read-only)"
run date date -u
run uname uname -a
run os-release cat /etc/os-release
run lscpu lscpu
run numa numactl -H
run meminfo free -h
run dmidecode-sudoless dmidecode -t bios
run hwloc-info hwloc-info
run hwloc-topo hwloc-ls
run lspci lspci -nn
run kfd-devs ls -l /dev/kfd /dev/dri 2>/dev/null
run kfd-sys sh -c 'for f in /sys/class/kfd/kfd*/gpu_id /sys/class/kfd/kfd*/name /sys/class/kfd/kfd*/unique_id; do echo "== $f"; cat "$f" 2>/dev/null; done'

echo "phase A: ROCm stack"
run rocm-smi-hw rocm-smi --showhw --json
run rocm-smi-product rocm-smi --showproductname --json
run rocm-smi-driver rocm-smi --showdriverversion --json
run rocm-smi-topo rocm-smi --showtopo
run rocm-smi-pids rocm-smi --showpids --json
run rocm-smi-mem rocm-smi --showmeminfo vram --json
run rocm-smi-power rocm-smi --showpower --json
run rocm-smi-clock rocm-smi --showclocks --json
run rocm-smi-temp rocm-smi --showtemp --json
run rocm-smi-busy rocm-smi --showbusy --json
run amd-smi-static amd-smi static
run amd-smi-metric amd-smi metric
run amd-smi-topo amd-smi topology
run rocminfo rocminfo
run hipconfig hipconfig --full
run hipcc hipcc --version
run rocm-vers cat /opt/rocm*/.info/version 2>/dev/null
run rocm-dirs ls -d /opt/rocm* /usr/lib/x86_64-linux-gnu/librccl* 2>/dev/null
run rocm-libs sh -c 'for d in /opt/rocm*/lib /opt/rocm*/lib64; do ls "$d"/libhipblaslt* "$d"/librccl* "$d"/librocwmma* "$d"/libMIOpen* "$d"/libamdhip* 2>/dev/null; done'
run ldconfig-rocm sh -c 'ldconfig -p 2>/dev/null | grep -Ei "rccl|hipblas|hip\.so|miopen|rocwmma|amdhip|composable"'
run llvm-tools sh -c 'for d in /opt/rocm*/llvm/bin; do ls "$d"/llvm-objdump "$d"/llvm-mc "$d"/clang 2>/dev/null; done'
run accel-sys ls /sys/class/accel 2>/dev/null
run driver-mod sh -c 'grep -H . /sys/module/amdgpu/version 2>/dev/null; lsmod | grep -E "amdgpu|amdkfd"'

echo "phase A: python environments"
PY_FOUND=""
for cand in "$EXTRA_PY" python3 python /opt/rocm*/bin/python3 $(ls /usr/bin/python3.* 2>/dev/null); do
  [ -x "$cand" ] || continue
  case "$cand" in *-config|*-config-*|[!/]*/python3.*-config) continue ;; esac
  v=$("$cand" -c 'import torch,sys;print(torch.__version__, (getattr(torch.version,"hip",None) or "cuda"))' 2>/dev/null || true)
  # only accept output that looks like a torch version line (python3.x-config
  # prints a usage banner and exits non-zero; without this check it is
  # misread as a torch env - seen live on amd-mi300x 2026-11-14)
  case "$v" in
    [0-9]*.[0-9]*) echo "  [.] torch found: $cand -> $v"; PY_FOUND="$cand"; break ;;
  esac
done
if [ -n "$PY_FOUND" ]; then
  "$PY_FOUND" -c 'import torch;print("torch",torch.__version__);print("hip",getattr(torch.version,"hip",None))' >"$BASE/logs/torch-env.txt" 2>&1
  "$PY_FOUND" -m pip list 2>/dev/null | grep -Ei 'torch|triton|numpy|nvidia|amd|rccl|transformers|safetensors' >>"$BASE/logs/torch-env.txt" || true
  "$PY_FOUND" -c 'import torch;print(torch.__config__.show())' >>"$BASE/logs/torch-env.txt" 2>&1 || true
else
  echo "  [!] no python with torch found — set MI300X_PYTHON=/path/to/python and rerun" | tee "$BASE/logs/torch-env.txt"
fi

echo "phase A: environment variables (ROCm-relevant)"
sh -c 'env | grep -Ei "ROCM|HIP|HSA|MIOPEN|NCCL|RCCL|GPU|TRITON" | sort' >"$BASE/logs/env.txt" 2>&1 || echo "__none__" >"$BASE/logs/env.txt"

# --- Phase B (capability probes, bounded single-GPU) and optional C ---
if [ -n "$PY_FOUND" ]; then
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  ARGS="--gpu $GPU --out $BASE/probes"
  [ "$BENCH" -eq 1 ] && ARGS="$ARGS --bench"
  echo "phase B: capability probes (bounded, GPU $GPU)"
  timeout 1500 "$PY_FOUND" "$SCRIPT_DIR/probes.py" cap $ARGS
  rc=$?
  echo "$rc" >"$BASE/probes/exit_code.txt"
  if [ "$rc" -ne 0 ]; then echo "  [!] probes exited $rc — logs in $BASE/probes/probe_results.json"; fi
else
  echo "phase B/C skipped: no torch python"
fi

# --- bundle ---
echo "bundling"
SUMMARY="$BASE/SUMMARY.txt"
{
  echo "MI300X evidence bundle $HOST $TS"
  echo "--- artifacts ---"; (cd "$BASE" && find . -type f | sort)
  echo "--- key facts ---"
  grep -m4 -E 'gfx|Marketing|Card series|GPU SKU' "$BASE/logs/rocminfo.txt" "$BASE/logs/rocm-smi-product.txt" 2>/dev/null || true
} >"$SUMMARY"
(cd "$BASE" && find . -type f -exec sha256sum {} \;) >"$BASE/SHA256SUMS.txt"
tar czf "$BASE.tar.gz" "$BASE"
echo "DONE -> $BASE.tar.gz"
echo "send back the whole .tar.gz (it contains logs/, probes/, SHA256SUMS.txt)"

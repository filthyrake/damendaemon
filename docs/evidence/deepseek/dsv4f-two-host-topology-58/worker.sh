#!/bin/bash
# #58 cell D: worker (stage [22,43)) on rocky10-desktop — CPU-only, 2x96 fleet plan.
set -e
cd /tmp/damendaemon-58-r10
export PYTHONPATH=/tmp/damendaemon-58-r10
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
exec /home/damen/3090-damendaemon/venv/bin/python -m scripts.run_deepseek_host_topology worker /tmp/dsv4f-58-snapshot \
  --plan /tmp/dsv4f-cell-d/plan.json --index 1 \
  --address 10.40.100.30 --port 9470 \
  --ca /tmp/dsv4f-cell-d/tls/ca.pem --cert /tmp/dsv4f-cell-d/tls/server.pem \
  --key /tmp/dsv4f-cell-d/tls/server.key --peer-cert /tmp/dsv4f-cell-d/tls/client.pem \
  --model-id 7727d9c10fff58a2d91929bee645c4e34d0591f3db31b207c67e4b4ef224b83c --verify-model-id \
  --identity-max-mib 160000 --identity-timeout 2400 \
  --context 128 --slots 2 --timeout 7200 --native-chat \
  --weight-mib 84000 --cache-mib 256 --output-mib 64 --frame-mib 64 \
  --prefill-chunk 8 --epoch 1

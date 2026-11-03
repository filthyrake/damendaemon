#!/bin/bash
# #58 cell D: coordinator (stage [0,22)) on labnode3 — completes a real chat turn.
set -e
cd /tmp/damendaemon-58
export PYTHONPATH=/tmp/damendaemon-58
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
exec /home/damen/dsv4f-venv/bin/python -m scripts.run_deepseek_host_topology complete /mnt/nas-models/models/DeepSeek-V4-Flash-0731 \
  --plan /tmp/dsv4f-cell-d/plan.json --index 0 --peers /tmp/dsv4f-cell-d/peers.json \
  --ca /tmp/dsv4f-cell-d/tls/ca.pem --cert /tmp/dsv4f-cell-d/tls/client.pem \
  --key /tmp/dsv4f-cell-d/tls/client.key \
  --model-id 7727d9c10fff58a2d91929bee645c4e34d0591f3db31b207c67e4b4ef224b83c --verify-model-id \
  --identity-max-mib 160000 --identity-timeout 2400 \
  --context 128 --slots 2 --timeout 7200 --native-chat \
  --weight-mib 84000 --cache-mib 256 --output-mib 64 --frame-mib 64 \
  --prefill-chunk 8 --epoch 1 \
  --prompt "Hello." --max-new-tokens 2

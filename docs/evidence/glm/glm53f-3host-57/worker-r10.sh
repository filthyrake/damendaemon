#!/bin/bash
# #57 3-host cell: worker stage [31,45) on rocky10-desktop — CPU-only, local full verified copy.
set -e
cd /tmp/damendaemon-g57-r10
export PYTHONPATH=/tmp/damendaemon-g57-r10
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
exec /home/damen/3090-damendaemon/venv/bin/python -m scripts.run_glm_flash_host_topology worker /home/damen/glm53f-snapshot \
  --plan /tmp/glm57-3host/plan.json --index 2 \
  --address 10.40.100.30 --port 9472 \
  --ca /tmp/glm57-3host/tls/ca.pem --cert /tmp/glm57-3host/tls/server-r10.pem \
  --key /tmp/glm57-3host/tls/server-r10.key --peer-cert /tmp/glm57-3host/tls/client.pem \
  --model-id 3160b6d537291fd28c80c4ec9f6c603bd57dddbfaefa1f218e4766637f635a88 --verify-model-id \
  --identity-max-mib 330000 --identity-timeout 3600 \
  --context 128 --slots 2 --timeout 7200 --native-chat \
  --weight-mib 112640 --cache-mib 4096 --workspace-mib 8192 --output-mib 64 --frame-mib 64 \
  --prefill-chunk 8 --epoch 1

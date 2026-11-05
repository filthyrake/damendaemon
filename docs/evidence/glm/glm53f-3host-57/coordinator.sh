#!/bin/bash
# #57 3-host cell: coordinator stage [0,17) on labnode3 — completes a real chat turn.
set -e
cd /tmp/damendaemon-g57
export PYTHONPATH=/tmp/damendaemon-g57
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
exec /home/damen/dsv4f-venv/bin/python -m scripts.run_glm_flash_host_topology complete /mnt/nas-models/models/GLM-5.3-Flash \
  --plan /tmp/glm57-3host/plan.json --index 0 --peers /tmp/glm57-3host/peers.json \
  --ca /tmp/glm57-3host/tls/ca.pem --cert /tmp/glm57-3host/tls/client.pem \
  --key /tmp/glm57-3host/tls/client.key \
  --model-id 3160b6d537291fd28c80c4ec9f6c603bd57dddbfaefa1f218e4766637f635a88 --verify-model-id \
  --identity-max-mib 330000 --identity-timeout 3600 \
  --context 128 --slots 2 --timeout 7200 --native-chat \
  --weight-mib 112640 --cache-mib 4096 --workspace-mib 8192 --output-mib 64 --frame-mib 64 \
  --prefill-chunk 8 --epoch 1 \
  --prompt "Hello." --max-new-tokens 2

#!/bin/bash
# #56 rung 5 3-host cell: coordinator stage [0,62) on labnode3 — completes a real native chat turn.
set -e
cd /tmp/damendaemon-g56
export PYTHONPATH=/tmp/damendaemon-g56
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
exec /home/damen/dsv4f-venv/bin/python -m scripts.run_glm_host_topology complete /mnt/nas-models/models/GLM-5.3 \
  --plan /tmp/glm56-3host/plan.json --index 0 --peers /tmp/glm56-3host/peers.json \
  --ca /tmp/glm56-3host/tls/ca.pem --cert /tmp/glm56-3host/tls/client.pem --key /tmp/glm56-3host/tls/client.key \
  --model-id 7d4d796bc284fd18b7fd4dd839695271eae11b9013d25fb9274f2aa7c15ba0dd --verify-model-id --identity-max-mib 760000 --identity-timeout 3600 \
  --context 128 --slots 2 --timeout 43200 --native-chat \
  --weight-mib 562375 --cache-mib 8192 --workspace-mib 128 --projection-workspace-mib 64 \
  --output-mib 64 --frame-mib 64 --prefill-chunk 8 --epoch 1 \
  --prompt "Hello." --max-new-tokens 2

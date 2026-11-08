#!/bin/bash
# #56 rung 5 3-host cell: worker stage [70,78) on 3090box — CPU-only, NAS original (final stage: tail group + norm + head).
set -e
cd /tmp/damendaemon-g56
export PYTHONPATH=/tmp/damendaemon-g56
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
exec /usr/bin/python3 -m scripts.run_glm_host_topology worker /mnt/nas-models/models/GLM-5.3 \
  --plan /tmp/glm56-3host/plan.json --index 2 \
  --address 10.30.108.121 --port 9472 \
  --ca /tmp/glm56-3host/tls/ca.pem --cert /tmp/glm56-3host/tls/server-3090.pem --key /tmp/glm56-3host/tls/server-3090.key \
  --peer-cert /tmp/glm56-3host/tls/client.pem \
  --model-id 7d4d796bc284fd18b7fd4dd839695271eae11b9013d25fb9274f2aa7c15ba0dd --verify-model-id --identity-max-mib 760000 --identity-timeout 3600 \
  --context 128 --slots 2 --timeout 43200 --native-chat \
  --weight-mib 80807 --cache-mib 8192 --workspace-mib 128 --projection-workspace-mib 64 \
  --output-mib 64 --frame-mib 64 --prefill-chunk 8 --epoch 1

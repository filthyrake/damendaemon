"""Serving-geometry reference for the 27B Q4 chain (#536 gate 4 cycle 8).

The cycle-7 reference built stages under the probe's max_context=512
admission clamp; the production chain-serving path (PR cycle 8) builds
stages natively at context 2048 via the chain opt-in. This runner generates
the matching-geometry in-process A30+A2 reference: same quantized path,
context 2048, production admission (chain opt-in), Triton stages.

Writes /tmp/dd536-27b-serve-protocol-reference.json (standard autoregressive protocol).
"""
import json
import os
import time
from pathlib import Path

import torch

os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN'] = '1'
os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
os.environ['DAMENDAEMON_HYBRID_PREFILL'] = 'scan'
os.environ['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = '22528,13312'
SEED = list(range(16, 48))
GREEDY = 12

from damendaemon.models import load as load_model
from damendaemon.hybrid_pipeline_build import build_cuda_pipeline

model = load_model('/tmp/q3827b', 2048, 'q4_0')
pipeline = build_cuda_pipeline(model, [0, 1], [22528 * 1024**2, 13312 * 1024**2], max_batch=1)
started = time.perf_counter()
pipeline.prefill(SEED)
prefill_s = time.perf_counter() - started
greedy, step_s = [], []
first = pipeline.prefill(SEED)
token = int(first.argmax(-1)[0])
greedy.append(token)
for _ in range(GREEDY - 1):
    mark = time.perf_counter()
    _, next_ids = pipeline.decode_batch([token], [0])
    token = int(next_ids[0])
    step_s.append(time.perf_counter() - mark)
    greedy.append(token)
mem = torch.cuda.max_memory_allocated() / 1024**3
report = {'mode': '27b-inprocess-a30a2-context2048-serving-protocol', 'greedy': greedy,
          'prefill_s': round(prefill_s, 3),
          'decode_steps_ms': [round(s * 1000, 1) for s in step_s],
          'max_gib': round(mem, 2), 'context': 2048,
          'admission': 'production chain opt-in', 'protocol': 'standard autoregressive (decode feeds the generated token, mirroring HostCompletion/ChainCompletion)'}
Path('/tmp/dd536-27b-serve-protocol-reference.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
pipeline.close()

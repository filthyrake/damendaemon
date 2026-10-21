"""#536 gate 4 cycle 10: 27B three-device capacity table (planner math).

Computes the choose_stage_splits storage plan for the qualified chain
topology budgets at contexts 2048 / 4096 / 8192 (batch 1): per-stage
layer bounds and required memory bytes (weights + KV/conv/state +
workspace). Pure CPU planning math on the pinned checkpoint — no GPU
allocation, no runtime claim. The live 8K attempt's observed per-stage
peaks are recorded alongside in the evidence (separate file).

Writes /tmp/dd536-27b-capacity.json.
"""
import json
import os
from pathlib import Path

os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN'] = '1'
os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
os.environ['DAMENDAEMON_HYBRID_PREFILL'] = 'scan'
os.environ['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = '22528,13312,22528'

BUDGETS_MIB = [22528, 13312, 22528]
CONTEXTS = [2048, 4096, 8192]

from damendaemon.models import load as load_model
from damendaemon.hybrid_layout import choose_stage_splits

model = load_model('/tmp/q3827b', 2048, 'q4_0')
table = {'checkpoint': '/tmp/q3827b', 'quantization': 'q4_0',
         'batch': 1, 'budgets_mib': BUDGETS_MIB,
         'contexts': {}}
for context in CONTEXTS:
    # The planner takes the context directly; the model loads once at the
    # admitted 2048 cell and the table extends the planning math only.
    placement = choose_stage_splits(model.weights, model.config, context, 1, 2,
                                    [mib * 1024**2 for mib in BUDGETS_MIB],
                                    scales=model.scales)
    stages = [dict(layer_start=b['layer_start'], layer_end=b['layer_end'],
                   required_gib=round(b['required_memory_bytes'] / 1024**3, 2))
              for b in placement['stages']]
    table['contexts'][str(context)] = {
        'storage_plan_found': bool(placement['storage_plan_found']),
        'reason': placement.get('reason'),
        'stages': stages,
        'total_required_gib': round(sum(s['required_gib'] for s in stages), 2),
    }
Path('/tmp/dd536-27b-capacity.json').write_text(json.dumps(table, indent=2))
print(json.dumps(table, indent=2))

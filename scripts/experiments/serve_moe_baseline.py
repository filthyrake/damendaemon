"""Validation-only server using an exact earlier OLMoE expert implementation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
from damendaemon.moe import OlmoeExecutor

if __name__ == '__main__':
    path = Path(os.environ['DAMENDAEMON_EXPERIMENT_MOE_BASELINE'])
    spec = importlib.util.spec_from_file_location('damendaemon.moe_baseline', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    method = os.environ.get('DAMENDAEMON_EXPERIMENT_MOE_METHOD', 'experts')
    if method not in ('experts', 'prefill_batch'):
        raise ValueError('unsupported OLMoE control method')
    setattr(OlmoeExecutor, method, getattr(baseline.OlmoeExecutor, method))
    Path(os.environ['DAMENDAEMON_EXPERIMENT_MOE_REPORT']).write_text(json.dumps({
        'scope':f'exact previous OLMoE {method} method; remaining runtime unchanged',
        'method':method,
        'source':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
    runpy.run_module('damendaemon', run_name='__main__')

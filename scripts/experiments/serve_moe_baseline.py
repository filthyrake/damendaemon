"""Validation-only server using an exact earlier OLMoE expert implementation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
from dllm.moe import OlmoeExecutor

if __name__ == '__main__':
    path = Path(os.environ['DLLM_EXPERIMENT_MOE_BASELINE'])
    spec = importlib.util.spec_from_file_location('dllm.moe_baseline', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    method = os.environ.get('DLLM_EXPERIMENT_MOE_METHOD', 'experts')
    if method not in ('experts', 'prefill_batch'):
        raise ValueError('unsupported OLMoE control method')
    setattr(OlmoeExecutor, method, getattr(baseline.OlmoeExecutor, method))
    Path(os.environ['DLLM_EXPERIMENT_MOE_REPORT']).write_text(json.dumps({
        'scope':f'exact previous OLMoE {method} method; remaining runtime unchanged',
        'method':method,
        'source':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
    runpy.run_module('dllm', run_name='__main__')

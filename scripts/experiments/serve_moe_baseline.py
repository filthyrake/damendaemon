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
    OlmoeExecutor.experts = baseline.OlmoeExecutor.experts
    Path(os.environ['DLLM_EXPERIMENT_MOE_REPORT']).write_text(json.dumps({
        'scope':'exact previous OLMoE experts method; remaining runtime unchanged',
        'source':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
    runpy.run_module('dllm', run_name='__main__')

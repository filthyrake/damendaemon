"""Explicit validation-only native AWQ server."""
import os
import runpy
from scripts.experiments.awq_marlin_dispatch import install_validation_dispatch

if __name__ == '__main__':
    install_validation_dispatch(os.environ['DLLM_EXPERIMENT_MARLIN_LIBRARY'])
    runpy.run_module('dllm', run_name='__main__')

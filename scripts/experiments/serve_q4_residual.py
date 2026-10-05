"""Explicit validation-only Q4 integer-dot server; never used by normal CLI."""
import runpy
from scripts.experiments.q4_residual import install_validation_dispatch

if __name__ == '__main__':
    install_validation_dispatch()
    runpy.run_module('dllm',run_name='__main__')

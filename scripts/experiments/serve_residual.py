"""Validation-only server entry point; never selected by the normal dllm CLI."""
import runpy
from scripts.experiments.kquant_residual import install_validation_dispatch


if __name__ == '__main__':
    install_validation_dispatch()
    runpy.run_module('dllm', run_name='__main__')

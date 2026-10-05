"""Explicit validation-only Q4 prompt tile server."""
import runpy
from scripts.experiments.q4_small_tiles import install_validation_dispatch

if __name__ == '__main__':
    install_validation_dispatch()
    runpy.run_module('dllm', run_name='__main__')

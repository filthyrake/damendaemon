"""Explicit idle-group admission experiment; normal CLI behavior is unchanged."""
import runpy
from scripts.experiments.batch_wait import install_validation_dispatch

if __name__=='__main__':
    install_validation_dispatch()
    runpy.run_module('dllm',run_name='__main__')

"""Explicit validation-only native AWQ server."""
import os
import runpy
from scripts.experiments.awq_marlin_dispatch import install_validation_dispatch

if __name__ == '__main__':
    install_validation_dispatch(os.environ['DAMENDAEMON_EXPERIMENT_MARLIN_LIBRARY'])
    runpy.run_module('damendaemon', run_name='__main__')

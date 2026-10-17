"""Explicit validation-only AWQ small-row decode server."""
import runpy
from scripts.experiments.awq_decode import install_validation_dispatch

if __name__ == '__main__':
    install_validation_dispatch()
    runpy.run_module('damendaemon', run_name='__main__')

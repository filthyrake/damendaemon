"""Validation-only server entry point; never selected by the normal damendaemon CLI."""
import runpy
from scripts.experiments.kquant_residual import install_validation_dispatch


if __name__ == '__main__':
    install_validation_dispatch()
    runpy.run_module('damendaemon', run_name='__main__')

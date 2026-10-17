"""Validation control for the previous Q4 prompt launch policy."""
import runpy
from damendaemon import quantized_kernels_q4

if __name__ == '__main__':
    quantized_kernels_q4.use_small_tile = lambda *args: False
    runpy.run_module('damendaemon', run_name='__main__')

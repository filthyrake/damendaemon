"""Validation-only FP32 residual server; never selected by the normal CLI."""
import runpy
import torch
from scripts.experiments.kquant_residual_f32 import install_validation_dispatch


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    install_validation_dispatch()
    runpy.run_module('damendaemon', run_name='__main__')

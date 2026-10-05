"""Validation-only compensated FP32 prefill server."""
import runpy
import torch
from scripts.experiments.kquant_prefill_f32 import install_validation_dispatch


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    install_validation_dispatch()
    runpy.run_module('dllm', run_name='__main__')

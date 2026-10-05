"""Validation server with explicit IEEE FP32 matrix-library settings."""
import runpy
import torch


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    runpy.run_module('dllm', run_name='__main__')

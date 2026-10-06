"""Diagnostic replay with reduced-precision GEMM accumulation disabled."""
import runpy
import sys
import torch

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
print('Diagnostic: FP16/BF16 reduced-precision reduction and TF32 disabled', file=sys.stderr)
runpy.run_path('scripts/replay_reference.py', run_name='__main__')

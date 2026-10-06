"""Bounded FP32 arithmetic control through the explicit runtime precision policy."""
import os
import runpy

os.environ['DLLM_HYBRID_PRECISION'] = 'float32'

if __name__ == '__main__':
    runpy.run_path('scripts/replay_reference.py', run_name='__main__')

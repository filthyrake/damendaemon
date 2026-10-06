"""Run the actual CLI via torchrun --run-path inside the owned job group.

Torchrun's subprocess/module mode starts separate worker sessions. Its Python
run-path mode uses multiprocessing workers that retain the harness process
group, so a deadline can terminate the launcher and both workers together.
"""
import runpy


if __name__ == '__main__':
    runpy.run_module('dllm', run_name='__main__', alter_sys=True)

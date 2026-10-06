"""Explicit Q4 Marlin validation server; no normal CLI option or admission claim."""
import os

from scripts.experiments.q4_marlin_dispatch import install_validation_dispatch


if __name__ == '__main__':
    install_validation_dispatch(os.environ['DLLM_EXPERIMENTAL_Q4_MARLIN_LIBRARY'])
    from dllm.qwen2 import Executor
    initialize = Executor.__init__
    def marked_initialize(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if getattr(getattr(self, 'quant_linear', None), 'experimental_q4_marlin_bytes', 0):
            self.kernel_backend += '+experimental-q4-marlin'
    Executor.__init__ = marked_initialize
    from dllm.cli import main
    main()

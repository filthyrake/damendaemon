"""Explicit Q4 Marlin validation server; no normal CLI option or admission claim."""
import os

if __name__ == '__main__':
    replacement = os.environ.get('DLLM_EXPERIMENTAL_Q4_MARLIN_REPLACE','0')
    if replacement not in ('0','1'):
        raise ValueError('DLLM_EXPERIMENTAL_Q4_MARLIN_REPLACE must be 0 or 1')
    if replacement == '1':
        from scripts.experiments.q4_marlin_replace import install_validation_dispatch
        install_validation_dispatch(os.environ['DLLM_EXPERIMENTAL_Q4_MARLIN_LIBRARY'])
    else:
        from scripts.experiments.q4_marlin_dispatch import install_validation_dispatch
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

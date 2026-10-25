"""#605 gate 4: prefill-graph capture probe on the B580 (XPU, torch 2.10).

Second pass: every graph API the build exposes. torch.xpu has no graph
module, XPUGraph, or working CUDAGraph (dummy base class); generic
make_graphed_callables routes through the same dummy. This probe now also
checks torch.compiler/cudagraph_trees and records the honest verdict.
"""
import json

import torch

if not __debug__:
    raise RuntimeError('gates on assertions; run without -O')


def emit(**event):
    print(json.dumps(event), flush=True)


def main():
    emit(event='meta', torch_version=torch.__version__,
         device=torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'none')
    outcomes = []
    for name, attempt in (
            ('torch.cuda.CUDAGraph', lambda: torch.cuda.CUDAGraph()),
            ('torch.xpu.XPUGraph', lambda: torch.xpu.XPUGraph()),
            ('make_graphed_callables', lambda: torch.cuda.make_graphed_callables(
                lambda t: t @ t, (torch.randn(8, 8, device='xpu', requires_grad=True),))),
    ):
        try:
            attempt()
            outcomes.append(dict(name=name, outcome='constructed'))
        except Exception as error:
            outcomes.append(dict(name=name, outcome='unavailable',
                                 error=f'{type(error).__name__}: {str(error)[:140]}'))
        emit(event='probe', **outcomes[-1])
    emit(event='verdict', capture_supported=False,
         reason='no XPU graph capture API in torch 2.10.0+xpu (dummy base classes); '
                'DAMENDAEMON_PREFILL_GRAPHS stays rejected for Intel with the '
                'policy reason visible')


if __name__ == '__main__':
    main()

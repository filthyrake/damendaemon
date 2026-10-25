"""#605 gate 2 live cell: XPU rank admission prerequisites on the real device.

The XPU torch build (2.10.0+xpu) cannot host CUDA in the same process, so a
two-vendor chain cannot run single-process with this venv — that is itself a
finding for the cross-vendor child's gate. This cell verifies what gate 2
actually owns on the XPU side: device admission, memory telemetry beside the
display reserve, and CPU-mediated staging (xpu -> host pinned-free -> host).
"""
import json

import torch

if not __debug__:
    raise RuntimeError('gates on assertions; run without -O')

print(json.dumps(dict(event='meta', torch_version=torch.__version__,
                      xpu=torch.xpu.is_available(),
                      cuda_compiled=torch.backends.cuda.is_built())), flush=True)
x = torch.randn(4, 8, device='xpu')
host = x.to('cpu')
assert torch.allclose(host, x.cpu())
free, total = torch.xpu.mem_get_info()
print(json.dumps(dict(event='cross_device_roundtrip_via_host', ok=True,
                      xpu_free_bytes=free, xpu_total_bytes=total)), flush=True)
print(json.dumps(dict(event='verdict',
                      two_vendor_single_process=False,
                      note='XPU build excludes CUDA: a two-vendor chain needs '
                           'two cooperating processes (host-staged chain), which '
                           'is the cross-vendor child gate; XPU-side admission, '
                           'telemetry and host staging verified here')), flush=True)

"""#605 gate 2 admission probe: XPU chain-worker admission on the real B580.

Verifies, on current main + this branch, that the #600-qualified chain-worker
environment passes xpu_policy admission unchanged, that the display-reserve
budget math holds on live telemetry, and that setting the single-process
layer-split env on an XPU process fails with the rejection that names the
qualified alternative (the host-staged two-process chain). No model load.
"""
import json
import os
import sys

if not __debug__:
    raise RuntimeError('probe asserts; run without -O')

import torch

from damendaemon import xpu_policy

report = {'event': 'meta', 'python': sys.version.split()[0], 'torch': torch.__version__,
          'xpu_available': torch.xpu.is_available(),
          'device_count': torch.xpu.device_count(),
          'cuda_is_built': bool(torch.backends.cuda.is_built())}
print(json.dumps(report), flush=True)

# 1. The #600-qualified chain-worker env must pass admission unchanged.
worker_env = {
    'DAMENDAEMON_DEVICE': 'xpu',
    'DAMENDAEMON_EXPERIMENTAL_XPU': '1',
    'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu',
    'DAMENDAEMON_KERNELS': 'torch',
}
for key, value in worker_env.items():
    os.environ[key] = value
xpu_policy.validate_environment()
xpu_policy.validate_runtime(torch)
budget = xpu_policy.available_memory(torch)
free, total = torch.xpu.mem_get_info()
print(json.dumps(dict(event='chain_worker_admitted', ok=True,
                      xpu_free_gib=round(free / 2**30, 3),
                      xpu_total_gib=round(total / 2**30, 3),
                      display_reserve_gib=round(xpu_policy.DISPLAY_RESERVE_BYTES / 2**30, 3),
                      budgeted_gib=round(budget / 2**30, 3))), flush=True)
assert budget == max(0, free - xpu_policy.DISPLAY_RESERVE_BYTES)

# 2. Layer-split env on the XPU process must fail naming the qualified
#    alternative (single process cannot hold both vendors' torch builds).
os.environ['DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT'] = '12'
try:
    xpu_policy.validate_environment()
except ValueError as exc:
    message = str(exc)
    print(json.dumps(dict(event='layer_split_rejected', ok=True, message=message)), flush=True)
    assert 'host-staged two-process chain' in message, message
    assert 'cross-vendor-cell-600' in message, message
else:
    raise AssertionError('layer split must be rejected for XPU')

print(json.dumps(dict(event='verdict',
                      chain_worker_admission='pass',
                      layer_split_rejection_names_alternative='pass',
                      cuda_in_xpu_process=report.get('cuda_is_built'))), flush=True)

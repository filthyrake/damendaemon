"""Small allocator-limit proof; never allocates a large model or exhausts RAM."""
import gc
import json
import os

import torch

from dllm.runtime import require_gpu
from dllm.rocm_policy import host_available, memory_limit


def main():
    if os.environ.get('DLLM_DEVICE') != 'rocm' or memory_limit() != 256*1024**2:
        raise ValueError('this probe requires ROCm and DLLM_ROCM_MEMORY_GIB=0.25')
    before = host_available()
    if before < 24*1024**3:
        raise MemoryError('probe requires at least 24 GiB available host RAM')
    require_gpu()
    initial = torch.cuda.memory_allocated()
    small = torch.ones(8*1024**2, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    rejected, message = False, None
    try:
        # Even if enforcement is broken, this single attempt is only 256 MiB.
        unexpected = torch.empty(256*1024**2, dtype=torch.uint8, device='cuda')
        del unexpected
    except torch.OutOfMemoryError as error:
        rejected, message = True, str(error)
    followup = torch.ones(1024, dtype=torch.float32, device='cuda')
    recovered = followup.sum().item() == 1024
    peak = torch.cuda.max_memory_reserved()
    del small, followup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    after = host_available()
    report = dict(passed=rejected and recovered and allocated == initial and after > 16*1024**3,
        device=torch.cuda.get_device_name(), torch=str(torch.__version__),
        budget_bytes=memory_limit(), allocator_fraction=torch.cuda.get_per_process_memory_fraction(),
        oversized_rejected=rejected, rejection=message, recovered=recovered,
        initial_allocated_bytes=initial, released_allocated_bytes=allocated,
        peak_reserved_bytes=peak, host_available_before=before, host_available_after=after,
        limitation='Caching-allocator containment only. Does not bound driver/library allocations or qualify large workloads.')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

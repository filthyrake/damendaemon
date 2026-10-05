"""Diagnostic only: identify cuBLAS workspace retention after a failed release gate.

The private PyTorch clear operation is never used by the runtime or accepted as
a passing lifecycle result. The original verifier still writes its failure.
"""
import json
from pathlib import Path
import sys

import torch

from verify_kquant_cache_lifecycle import main


if __name__ == '__main__':
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        main()
    except AssertionError as error:
        before = torch.cuda.memory_allocated()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        output.with_suffix('.workspace.json').write_text(json.dumps(dict(
            lifecycle_passed=False, failure=repr(error), allocated_before_clear=before,
            allocated_after_clear=after, released_by_cublas_clear=before-after,
            diagnostic_only=True), indent=2) + '\n')
        raise

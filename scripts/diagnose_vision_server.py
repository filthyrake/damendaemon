"""Validation-only CUDA allocation history for the image HTTP lifecycle."""
import json
from pathlib import Path
import torch
from damendaemon.runtime import Runtime
from damendaemon.cli import main

torch.cuda.memory._record_memory_history(max_entries=20000)
original=Runtime._status


def status(self):
    result=original(self)
    if result['host_cached_bytes']==0 and result['cuda_allocated_bytes']:
        snapshot=torch.cuda.memory._snapshot()
        active=[block for segment in snapshot['segments'] for block in segment['blocks']
                if block['state']=='active_allocated']
        Path('results/qwen35-http-retained-allocations.json').write_text(json.dumps(active,indent=2)+'\n')
        if all(any('getNewWorkspace' in frame.get('name','') for frame in block.get('frames',[])) for block in active):
            result['diagnostic_cublas_workspace_bytes']=sum(block['size'] for block in active)
    return result


Runtime._status=status
main()

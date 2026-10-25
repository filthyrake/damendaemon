"""Validation-only ROCm allocation history for the image HTTP lifecycle (#50).

Mirrors diagnose_vision_server.py for the HIP path: records allocator
history, snapshots live allocations at the unload status (host cache empty),
and classifies every live block by its allocation frames. The persistent
hipBLAS workspace vocabulary on this build is recorded as observed; blocks
whose frames identify no library entrypoint are listed honestly instead of
being waved through.
"""
import json
from pathlib import Path
import torch
from damendaemon.runtime import Runtime
from damendaemon.cli import main

torch.cuda.memory._record_memory_history(max_entries=20000)
original = Runtime._status


def status(self):
    result = original(self)
    if result['host_cached_bytes'] == 0 and result['hip_allocated_bytes']:
        snapshot = torch.cuda.memory._snapshot()
        active = [block for segment in snapshot['segments'] for block in segment['blocks']
                  if block['state'] == 'active_allocated']
        Path('/tmp/v50/qwen35-vision-homer-retained-allocations.json').write_text(
            json.dumps(active, indent=2) + '\n')
        identified = [block for block in active
                      if any('Workspace' in frame.get('name', '')
                             or 'getNewWorkspace' in frame.get('name', '')
                             for frame in block.get('frames', []))]
        unidentified = [block for block in active if block not in identified]
        if not unidentified:
            result['diagnostic_hipblas_workspace_bytes'] = sum(block['size'] for block in identified)
        else:
            result['diagnostic_unidentified_allocations'] = [
                dict(size=block['size'], frames=[frame.get('name', '')[:80]
                                                 for frame in block.get('frames', [])[:4]])
                for block in unidentified]
    return result


Runtime._status = status
main()

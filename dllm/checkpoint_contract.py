"""Shared bounded metadata-only checkpoint audit plumbing."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    storage: str


def audit_snapshot(path,describe,audit_names,audit_headers,*,names_only=False):
    from .checkpoint import _unique_fields,inspect_snapshot
    from .download import INDEX,snapshot_files
    path=Path(path)
    def read(filename,limit):
        with (path/filename).open('rb') as reader:value=reader.read(limit+1)
        if len(value)>limit:raise ValueError('checkpoint metadata exceeds audit size limit')
        return json.loads(value,object_pairs_hook=_unique_fields)
    config=read('config.json',1024*1024)
    describe(config) # reject architecture/inventory bounds before touching shards
    if names_only:
        index=read(INDEX,16*1024*1024)
        if (not isinstance(index,dict) or not isinstance(index.get('weight_map'),dict)
                or any(not isinstance(value,str) for value in index['weight_map'].values())):
            raise ValueError('checkpoint index requires string shard paths')
        snapshot_files([INDEX,*index['weight_map'].values()],index)
        report=audit_names(config,index['weight_map'])
    else:
        _,inventory=inspect_snapshot(path)
        headers={name:header for shard in inventory.values() for name,header in shard['tensors'].items()}
        report=audit_headers(config,headers)
    return dict(scope='tensor_names_only' if names_only else 'tensor_header_contract',
                main_tensors=report['main_tensors'],excluded_mtp_count=len(report['excluded_mtp']),
                mtp_validated=False,weight_values_read=False,model_execution_qualified=False)

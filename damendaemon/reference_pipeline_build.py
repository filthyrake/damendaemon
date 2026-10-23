"""Global byte admission and file receipts for explicit CPU checkpoint pipelines."""
from pathlib import Path
import stat
from .checkpoint import snapshot_shards
from .network_filesystem import network_device


def snapshot_inventory(path):
    files,index=snapshot_shards(path)
    if len(files)>256:raise ValueError('reference pipeline checkpoint exceeds shard limit')
    names=[path/'config.json',*files]
    if index is not None:names.append(path/'model.safetensors.index.json')
    result=[]
    for file in names:
        info=file.stat()
        if not stat.S_ISREG(info.st_mode):raise ValueError('reference pipeline requires regular checkpoint files')
        # Network filesystems may jitter ctime on reads; keep the strict
        # receipt for local checkpoint directories.
        if network_device(info.st_dev):
            result.append((str(file),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns))
        else:
            result.append((str(file),info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns))
    return tuple(result)


def build_reference_pipeline(path,layer_splits,*,inspect_snapshot,ranges_for,load_stage,pipeline_type,
                             options,inventory=snapshot_inventory):
    """Family callbacks are explicit; global inspection must not read payloads.

    Receipts detect ordinary concurrent replacement/modification, not hostile
    writers or authenticated model bytes. No transport, model registration or
    generation ownership is implied by checkpoint construction.
    """
    if (not isinstance(layer_splits,(list,tuple)) or not 1<=len(layer_splits)<=7
            or any(type(value) is not int or not 1<=value<128 for value in layer_splits)
            or list(layer_splits)!=sorted(set(layer_splits))):
        raise ValueError('reference pipeline needs 1..7 ordered distinct internal layer splits')
    path=Path(path).expanduser().resolve();before=inventory(path)
    config,retained,overlap=inspect_snapshot(path,**options)
    ranges=ranges_for(config,layer_splits)
    if inventory(path)!=before:raise ValueError('reference pipeline checkpoint changed during admission')
    stages=[]
    try:
        for start,end in ranges:
            if inventory(path)!=before:raise ValueError('reference pipeline checkpoint changed before stage loading')
            stage=load_stage(path,start,end,**options)
            if stage.config!=config or inventory(path)!=before:
                raise ValueError('reference pipeline checkpoint changed during stage loading')
            stages.append(stage)
        pipeline=pipeline_type(stages)
        if inventory(path)!=before:raise ValueError('reference pipeline checkpoint changed during construction')
    except BaseException:
        stages.clear();stage=None
        raise
    pipeline.retained_weight_bytes=retained;pipeline.weight_overlap_bytes=overlap
    return pipeline

"""Bounded visual-only Flash snapshot loading for the explicit CPU reference."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import torch
from .checkpoint import snapshot_shards
from .safetensor_header import read_header,file_stamp,ITEM_BYTES,MAX_HEADER_BYTES,_unique_fields
from .packed_checkpoint import _payload
from .glm_flash_vision import FlashVision,vision_specs,audit_vision,ROOT


def load_reference_vision(path,*,max_weight_bytes=256*1024**2,max_tokens=4096,
                          max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Read only admitted visual values, preserving strict whole-index consistency.

    Source FP16/BF16/F32 tensors become owned CPU FP32 values. Weight admission
    includes all retained values and the largest source conversion overlap.
    Metadata/Python overhead is separate: at most 256 descriptors, 64 MiB total
    headers, 16 MiB per header and 1 MiB configuration. No official weight,
    trained-model semantic or accelerator qualification follows from this loader.
    """
    if sys.byteorder!='little':raise ValueError('Flash vision loading requires little-endian storage')
    if (type(max_tokens) is not int or not 1<=max_tokens<=65536
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_weight_bytes,max_workspace_bytes,max_output_bytes))):
        raise ValueError('invalid Flash visual loader budgets')
    path=Path(path)
    with (path/'config.json').open('rb') as reader:raw=reader.read(1024**2+1)
    if len(raw)>1024**2:raise ValueError('Flash visual configuration exceeds metadata limit')
    config=json.loads(raw,object_pairs_hook=_unique_fields)
    if not isinstance(config,dict):raise ValueError('expected Flash root configuration')
    specs=vision_specs(config.get('vision_config'))
    # Validate exclusions/text width before opening any checkpoint descriptor.
    audit_vision(config,[ROOT+name for name in specs])
    files,index=snapshot_shards(path)
    if len(files)>256:raise ValueError('Flash visual checkpoint exceeds descriptor limit')
    weights={}
    with ExitStack() as stack:
        selected={};actual={};opened=[];header_bytes=0
        for file in files:
            reader=stack.enter_context(file.open('rb'))
            entries,base,stamp=read_header(reader,max_header_bytes=min(MAX_HEADER_BYTES,64*1024**2-header_bytes))
            header_bytes+=base-8;opened.append((reader,stamp))
            for name,entry in entries.items():
                if name in actual:raise ValueError('duplicate Flash visual snapshot tensor')
                actual[name]=str(file.relative_to(path))
                if name.startswith(ROOT):selected[name]=(reader,base,entry)
        if index is not None and actual!=index['weight_map']:raise ValueError('Flash visual headers disagree with shard index')
        audit_vision(config,actual,{name:entry for name,(_,_,entry) in selected.items()})
        retained=sum(math.prod(shape)*4 for shape in specs.values())
        largest=max(math.prod(entry['shape'])*ITEM_BYTES[entry['dtype']] for _,_,entry in selected.values())
        if retained+largest>max_weight_bytes:raise ValueError('Flash visual parameters exceed aggregate weight budget')
        if 16*max(math.prod(shape) for shape in specs.values())>max_workspace_bytes:
            raise ValueError('Flash visual parameter validation exceeds workspace budget')
        for name,shape in specs.items():
            reader,base,entry=selected[ROOT+name];size=entry['data_offsets'][1]-entry['data_offsets'][0]
            raw=_payload(reader,base,entry|{'shape':[size]})
            dtype={'F16':torch.float16,'BF16':torch.bfloat16,'F32':torch.float32}[entry['dtype']]
            value=raw.view(dtype).reshape(shape).float()
            if not torch.isfinite(value).all():raise ValueError('nonfinite Flash visual checkpoint parameter')
            weights[name]=value
            del raw,value
        if any(file_stamp(reader)!=stamp for reader,stamp in opened):raise ValueError('Flash visual checkpoint changed during reads')
    result=FlashVision(config['vision_config'],weights,max_weight_bytes=max_weight_bytes,max_tokens=max_tokens,
                       max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_output_bytes)
    result.source_config=config
    return result

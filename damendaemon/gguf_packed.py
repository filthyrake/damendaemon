"""Host validation and compact layout descriptions for native GGUF block rows."""
import numpy as np
import torch

FORMATS = {2:(32,18), 8:(32,34), 12:(256,144), 13:(256,176), 14:(256,210)}
KQUANTS = frozenset((12,13,14))
NAMES = {2:'q4_0', 8:'q8_0', 12:'q4_k', 13:'q5_k', 14:'q6_k'}


def read_raw(file, info, embedding=False):
    block,size = FORMATS[info.kind]
    file.seek(info.offset)
    raw=np.frombuffer(file.read(info.length),dtype=np.uint8).copy()
    if raw.nbytes != info.length:
        raise ValueError(f'truncated GGUF tensor: {info.name}')
    blocks=raw.reshape(-1,size)
    headers=(blocks[:,:4] if info.kind in (12,13) else blocks[:,208:210]
             if info.kind==14 else blocks[:,:2]).copy().view('<f2')
    if not np.isfinite(headers).all():
        raise ValueError(f'GGUF tensor has non-finite scales: {info.name}')
    if embedding:
        # Conservative bound avoids expanding ordinary embedding blocks merely
        # to check finiteness. Unusual large scales use bounded exact decoding.
        bound={2:8,8:128,12:1008,13:2016,14:4096}[info.kind]
        if np.abs(headers.astype(np.float32)).max() > np.finfo(np.float16).max/bound:
            from gguf import GGMLQuantizationType
            from gguf.quants import dequantize
            per_row=info.shape[1]//block
            for first in range(0,len(blocks),128*per_row):
                values=dequantize(blocks[first:first+128*per_row],GGMLQuantizationType(info.kind))
                with np.errstate(over='ignore'):
                    finite=np.isfinite(values.astype(np.float16)).all()
                if not finite:raise ValueError('GGUF embedding exceeds finite FP16 range')
    return torch.from_numpy(raw.reshape(info.shape[0],-1))


def layout(width, parts):
    """Describe contiguous row segments as (first row, last row, type, byte offset)."""
    if type(width) is not int or width<=0:
        raise ValueError('packed matrix width must be positive')
    spec=[]; first=0; offset=0
    for tensor,kind in parts:
        if tensor.dtype!=torch.uint8 or tensor.ndim!=2 or not tensor.is_contiguous() or not len(tensor):
            raise ValueError('packed matrix segments must be contiguous byte rows')
        if type(kind) is not int or kind not in (1,*FORMATS):
            raise ValueError('unsupported packed matrix segment format')
        block,size=(1,2) if kind==1 else FORMATS[kind]
        if width%block or tensor.shape[1]!=width//block*size:
            raise ValueError('packed matrix segment shape disagrees with its format')
        rows=len(tensor)
        if spec and spec[-1][2]==kind:
            previous=spec[-1]
            spec[-1]=(previous[0],first+rows,kind,previous[3])
        else:
            spec.append((first,first+rows,kind,offset))
        first+=rows;offset+=tensor.numel()
    if not spec:raise ValueError('packed matrix needs at least one segment')
    return width,tuple(spec)

"""Bounded raw packed safetensors reads, including E8M0 byte storage.

This is a selected-matrix CPU loader, not whole-model admission. Raw byte
weights and E8M0 byte or explicit FP32 block scales are returned; no weight expansion or device allocation occurs.
"""
from contextlib import ExitStack
from pathlib import Path
import sys

import torch

from .safetensor_header import MAX_HEADER_BYTES, read_header as _header, file_stamp as _stamp
from .packed_moe import PackedMatrix



def _payload(reader,base,entry):
    start,stop=entry['data_offsets']
    buffer=bytearray(stop-start)
    reader.seek(base+start)
    view=memoryview(buffer)
    count=0
    while count < len(buffer):
        read=reader.readinto(view[count:])
        if not read:
            raise ValueError('packed checkpoint payload was truncated during read')
        count+=read
    return torch.frombuffer(buffer,dtype=torch.uint8).reshape(entry['shape'])


def read_packed_matrix(weight_path, weight_name, scale_path, scale_name, *, format,
                       max_bytes=64*1024*1024):
    """Copy exactly one packed matrix and its scale bytes under a joint budget.

    Paths may name the same shard or different shards. Full headers and payload
    range coverage are checked, then both selected shapes/types and combined byte
    size are validated before either value payload is read. Headers are separately
    capped at 16 MiB per shard; the budget covers retained raw tensor payloads,
    excluding header/Python/allocator overhead. Caller owns the returned buffers.
    File descriptors close on success and every failure. Files must stay unchanged
    while reading; mutation detected by descriptor metadata is rejected.
    """
    if format not in ('e2m1','e4m3fn','e4m3fn_f32') or type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('invalid packed checkpoint format or byte budget')
    if format == 'e4m3fn_f32' and sys.byteorder != 'little':
        raise ValueError('FP32 packed scales require a little-endian host')
    if any(not isinstance(name,str) or not name or name == '__metadata__' for name in (weight_name,scale_name)):
        raise ValueError('packed checkpoint tensor names must be nonempty')
    with ExitStack() as stack:
        opened={}
        selected=[]
        for path,name in ((weight_path,weight_name),(scale_path,scale_name)):
            path=Path(path).resolve()
            if path not in opened:
                reader=stack.enter_context(path.open('rb'))
                entries,base,stamp=_header(reader)
                opened[path]=(reader,entries,base,stamp)
            reader,entries,base,_=opened[path]
            if name not in entries:
                raise ValueError('packed checkpoint tensor is missing')
            selected.append((reader,base,entries[name]))
        weight,scale=(item[2] for item in selected)
        expected_dtype=('I8','U8') if format == 'e2m1' else ('F8_E4M3','F8_E4M3FN')
        scale_dtypes=('F32',) if format == 'e4m3fn_f32' else ('F8_E8M0','F8_E8M0FNU')
        if (weight['dtype'] not in expected_dtype or scale['dtype'] not in scale_dtypes
                or len(weight['shape']) != 2 or min(weight['shape']) < 1):
            raise ValueError('packed checkpoint weight/scale storage does not match format')
        rows,columns=weight['shape']
        if format == 'e2m1':columns*=2
        scale_shape=[rows,columns//32] if format == 'e2m1' else [(rows+127)//128,(columns+127)//128]
        if (format == 'e2m1' and columns%32) or scale['shape'] != scale_shape:
            raise ValueError('packed checkpoint scale geometry does not match weight')
        needed=sum(entry['data_offsets'][1]-entry['data_offsets'][0] for _,_,entry in selected)
        if needed > max_bytes:
            raise ValueError('packed checkpoint pair exceeds raw byte budget')
        values=[]
        for index,(reader,base,entry) in enumerate(selected):
            if index == 1 and format == 'e4m3fn_f32':
                size=entry['data_offsets'][1]-entry['data_offsets'][0]
                raw=_payload(reader,base,entry|{'shape':[size]})
                values.append(raw.view(torch.float32).reshape(entry['shape']))
            else:values.append(_payload(reader,base,entry))
        if any(_stamp(reader) != stamp for reader,_,_,stamp in opened.values()):
            raise ValueError('packed checkpoint changed while reading values')
        return PackedMatrix(*values,format)

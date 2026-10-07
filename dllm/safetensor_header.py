"""Strict bounded safetensors metadata validation without tensor libraries."""
import json
import math
import os
import struct

def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate checkpoint JSON field')
        result[name] = value
    return result


MAX_HEADER_BYTES = 16*1024*1024
ITEM_BYTES = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E4M3FN':1,'F8_E5M2':1,
              'F8_E8M0':1,'F8_E8M0FNU':1,'I16':2,'U16':2,'F16':2,'BF16':2,
              'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}


def file_stamp(reader):
    state=os.fstat(reader.fileno())
    return state.st_size,state.st_mtime_ns,state.st_ctime_ns


def read_header(reader, *, max_header_bytes=MAX_HEADER_BYTES):
    if type(max_header_bytes) is not int or not 2 <= max_header_bytes <= MAX_HEADER_BYTES:
        raise ValueError('invalid remaining checkpoint header budget')
    before=file_stamp(reader)
    prefix=reader.read(8)
    if len(prefix) != 8:
        raise ValueError('truncated packed checkpoint header length')
    length=struct.unpack('<Q',prefix)[0]
    if not 2 <= length <= max_header_bytes or 8+length > before[0]:
        raise ValueError('packed checkpoint header exceeds bounds')
    raw=reader.read(length)
    try:
        header=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique_fields)
    except (ValueError,UnicodeError) as error:
        raise ValueError('invalid packed checkpoint JSON header') from error
    if not isinstance(header,dict) or len(header) > 100001:
        raise ValueError('invalid packed checkpoint tensor inventory')
    data_bytes=before[0]-8-length
    ranges=[]
    tensors={}
    for name,entry in header.items():
        if name == '__metadata__':
            if not isinstance(entry,dict) or any(not isinstance(v,str) for v in entry.values()):
                raise ValueError('invalid packed checkpoint metadata')
            continue
        if (not name or not isinstance(entry,dict) or set(entry) != {'dtype','shape','data_offsets'}
                or not isinstance(entry['dtype'],str) or entry['dtype'] not in ITEM_BYTES):
            raise ValueError('unsupported packed checkpoint tensor metadata')
        shape,offsets=entry['shape'],entry['data_offsets']
        if (not isinstance(shape,list) or len(shape) > 16
                or any(type(v) is not int or not 0 <= v < 2**63 for v in shape)
                or not isinstance(offsets,list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_bytes
                or offsets[1]-offsets[0] != math.prod(shape)*ITEM_BYTES[entry['dtype']]):
            raise ValueError('invalid packed checkpoint tensor shape/offsets')
        ranges.append(tuple(offsets))
        tensors[name]=entry
    cursor=0
    for start,stop in sorted(ranges):
        if start != cursor:
            raise ValueError('packed checkpoint payload has overlapping or missing ranges')
        cursor=stop
    if cursor != data_bytes or file_stamp(reader) != before:
        raise ValueError('packed checkpoint payload is incomplete or changed')
    return tensors,8+length,before

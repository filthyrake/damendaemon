"""Bounded access to the MIT-licensed gguf reader; no inference engine dependency.

Only the little-endian scalar/vector/matrix formats consumed by dllm are
accepted. Validate sizes with Python integers before upstream NumPy arithmetic.
"""
import math
from pathlib import Path
import struct

import numpy as np

try:
    from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType
except ImportError as error:
    raise ImportError('GGUF support requires the optional dllm[gguf] dependency') from error


MAX_HEADER = 256 * 1024 * 1024
MAX_ITEMS = 1_000_000
from .gguf_packed import FORMATS as PACKED_FORMATS
FORMATS = {0: (1, 4), 1: (1, 2), **PACKED_FORMATS}


class BoundedReader(GGUFReader):
    def __init__(self, path, *, allow_empty=False):
        with Path(path).open('rb') as file:
            header = file.read(24)
        if len(header) != 24:
            raise ValueError('truncated GGUF header')
        magic, version, tensors, fields = struct.unpack('<4sIQQ', header)
        if magic != b'GGUF' or version not in (2, 3):
            raise ValueError('expected little-endian GGUF version 2 or 3')
        if type(allow_empty) is not bool or not (0 if allow_empty else 1) <= tensors <= 100_000 or fields > 100_000:
            raise ValueError('GGUF tensor or metadata count exceeds supported bounds')
        self._header_phase = True
        try:
            super().__init__(path, mode='r')
        except (IndexError, KeyError, OverflowError, UnicodeError) as error:
            self.close()
            raise ValueError('malformed GGUF metadata') from error
        except BaseException:
            self.close();raise

    def close(self):
        mapping=getattr(getattr(self,'data',None),'_mmap',None)
        if mapping is not None and not mapping.closed:mapping.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def _get(self, offset, dtype, count=1, override_order=None):
        offset, count = int(offset), int(count)
        end = offset + np.dtype(dtype).itemsize * count
        limit = min(len(self.data), MAX_HEADER) if self._header_phase else len(self.data)
        if offset < 0 or count < 0 or end > limit:
            raise ValueError('GGUF read exceeds file or metadata bounds')
        # Plain ndarray views avoid cascading np.memmap slicing/view objects
        # for every token string. The supported file byte order is little-endian.
        return np.frombuffer(self.data, dtype=np.dtype(dtype).newbyteorder('<'),
                             count=count, offset=offset)

    def _get_str(self, offset):
        length = int(self._get(offset, np.uint64)[0])
        if length > 16 * 1024 * 1024:
            raise ValueError('GGUF string exceeds supported bounds')
        return self._get(offset, np.uint64), self._get(offset + 8, np.uint8, length)

    def _get_field_parts(self, orig_offs, raw_type):
        if raw_type == GGUFValueType.ARRAY:
            subtype = int(self._get(orig_offs, np.uint32)[0])
            count = int(self._get(orig_offs + 4, np.uint64)[0])
            if subtype == GGUFValueType.ARRAY or count > MAX_ITEMS:
                raise ValueError('nested or oversized GGUF metadata array')
        return super()._get_field_parts(orig_offs, raw_type)

    def _build_tensors(self, start_offs, fields):
        if start_offs > MAX_HEADER or start_offs > len(self.data):
            raise ValueError('GGUF tensor data starts outside supported header bounds')
        intervals = []
        for field in fields:
            _, _, _, dims, raw_type, offset = field.parts
            shape = tuple(int(n) for n in dims)
            kind = int(raw_type[0])
            if len(shape) not in (1, 2) or any(n <= 0 for n in shape):
                raise ValueError('GGUF weights must be nonempty vectors or matrices')
            if kind not in FORMATS:
                raise ValueError(f'unsupported GGUF weight type: {GGMLQuantizationType(kind).name}')
            block, size = FORMATS[kind]
            if shape[0] % block or (block > 1 and len(shape) != 2):
                raise ValueError(f'GGUF packed matrices require rows divisible by {block}')
            length = math.prod(shape) // block * size
            first = int(start_offs) + int(offset[0])
            last = first + length
            if int(offset[0]) % int(self.alignment) or first < start_offs or last > len(self.data):
                raise ValueError('GGUF tensor has an unaligned or out-of-bounds byte range')
            intervals.append((first, last))
        intervals.sort()
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError('GGUF tensor byte ranges overlap')
        self._header_phase = False
        super()._build_tensors(start_offs, fields)

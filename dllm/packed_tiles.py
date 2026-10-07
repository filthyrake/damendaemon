"""Bounded CPU checkpoint diagnostics, not a runtime packed projection.

Independent byte decoding follows OCP MX v1.0 E2M1/E4M3/E8M0 semantics.
Tile scale layouts follow DeepSeek-V4-Flash inference/model.py (MIT), revision
60d8d70770c6776ff598c94bb586a859a38244f1. Canonical Q4_0/Q8_0 integer
formats use FP16 per-row/32-column scales; Q8 storage is signed int8.
"""
import torch

MAX_TILE_ELEMENTS = 65536


def _scale(raw):
    # Byte zero is 2**-127, not IEEE exponent-zero's zero value.
    code = raw.to(torch.int32)
    bits = torch.where(code == 0, 0x00400000, code << 23)
    bits = torch.where(code == 255, 0x7fc00000, bits)
    return bits.contiguous().view(torch.float32)


def _e2m1(raw):
    code = raw.to(torch.int32)
    unsigned = code & 7
    magnitude = torch.where(unsigned < 4, unsigned.float()*.5,
                            torch.where(unsigned < 6, unsigned.float()-2, 2*unsigned.float()-8))
    return torch.where((code & 8) != 0, -magnitude, magnitude)


def _e4m3(raw):
    code = raw.to(torch.int32)
    exponent, fraction = (code >> 3) & 15, code & 7
    magnitude = torch.where(exponent == 0, fraction.float()*2**-9,
                            torch.ldexp(1+fraction.float()/8, exponent-7))
    magnitude = torch.where((exponent == 15) & (fraction == 7), float('nan'), magnitude)
    return torch.where((code & 128) != 0, -magnitude, magnitude)


def validate_layout(weight, scales, *, format):
    """Validate raw CPU storage metadata without decoding or reading values."""
    formats=('e2m1','e4m3fn','e4m3fn_f32','q4_0_f16','q8_0_f16')
    if format not in formats:raise ValueError('unsupported packed tile format')
    scale_dtype={'e4m3fn_f32':torch.float32,'q4_0_f16':torch.float16,'q8_0_f16':torch.float16}.get(format,torch.uint8)
    weight_dtype=torch.int8 if format=='q8_0_f16' else torch.uint8
    if (not isinstance(weight,torch.Tensor) or weight.dtype!=weight_dtype or weight.device.type!='cpu'
            or not isinstance(scales,torch.Tensor) or scales.dtype!=scale_dtype or scales.device.type!='cpu'
            or weight.ndim!=2 or scales.ndim!=2 or min(weight.shape)<1):
        raise ValueError('packed tile inputs require matching typed CPU weights and scales')
    rows,packed_columns=weight.shape
    columns=packed_columns*2 if format in ('e2m1','q4_0_f16') else packed_columns
    row_groups=format in ('e2m1','q4_0_f16','q8_0_f16')
    expected=(rows,columns//32) if row_groups else ((rows+127)//128,(columns+127)//128)
    if (row_groups and columns%32) or scales.shape!=expected:
        raise ValueError('packed weight shape and scale layout do not agree')
    return rows, columns


def decode_tile(weight, scales, *, format, row_start, row_stop, column_start, column_stop):
    """Decode at most 65,536 values into FP32, preserving format special values.

    Formats: 'e2m1' has low-nibble-first pairs and row/32-column E8M0 scales;
    'e4m3fn' has byte elements and 128x128-block E8M0 scales. Both inputs must
    remain raw uint8 CPU matrices for those formats. Explicit e4m3fn_f32 uses
    FP32 128x128 block scales instead. Ranges are nonempty, half-open logical axes.
    q4_0_f16 uses canonical low/high half-block nibbles minus eight, while
    q8_0_f16 uses signed bytes; both use FP16 row/32-column scales.
    No full-input cast, scan or copy occurs; only the requested tile is expanded.
    NaN scale bytes propagate; finite products may overflow FP32.
    """
    rows, columns = validate_layout(weight, scales, format=format)
    bounds = (row_start, row_stop, column_start, column_stop)
    if (any(type(value) is not int for value in bounds)
            or not 0 <= row_start < row_stop <= rows
            or not 0 <= column_start < column_stop <= columns):
        raise ValueError('tile bounds must be nonempty integer ranges within logical weight axes')
    if (row_stop-row_start)*(column_stop-column_start) > MAX_TILE_ELEMENTS:
        raise ValueError('packed checkpoint tile exceeds 65536 decoded elements')
    column_ids = torch.arange(column_start, column_stop, device='cpu')
    row_ids = torch.arange(row_start, row_stop, device='cpu')
    if format == 'e2m1':
        packed = weight[row_start:row_stop, column_ids//2]
        codes = (packed >> ((column_ids % 2)*4)) & 15
        values = _e2m1(codes)
        scale_bytes = scales[row_start:row_stop, column_ids//32]
    elif format in ('q4_0_f16','q8_0_f16'):
        if format=='q4_0_f16':
            packed=weight[row_start:row_stop,(column_ids//32)*16+column_ids%16]
            values=((packed >> ((column_ids%32)//16*4)) & 15).float()-8
        else:values=weight[row_start:row_stop,column_start:column_stop].float()
        return values*scales[row_start:row_stop,column_ids//32].float()
    else:
        values = _e4m3(weight[row_start:row_stop, column_start:column_stop])
        scale_bytes = scales[row_ids[:, None]//128, column_ids[None, :]//128]
    return values*(scale_bytes if format == 'e4m3fn_f32' else _scale(scale_bytes))

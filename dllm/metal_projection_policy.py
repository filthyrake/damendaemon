"""Measured short-prefill dispatch for native Metal quantized projections."""


def use_packed_projection(rows, outputs, width, fp32):
    if rows <= 4:
        return True
    # Additional prompt rows are qualified only for FP32 activations and layouts
    # whose element offsets fit the accelerated unsigned-index shader.
    if not fp32 or max(outputs*width, rows*width, rows*outputs) >= 2**32:
        return False
    return rows <= 8 or (rows <= 16 and outputs*width >= 4_000_000)

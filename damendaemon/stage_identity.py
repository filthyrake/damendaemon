"""Content identity for prepared CPU weights before stage-group activation."""
import ctypes
import hashlib
import json
import sys


def fingerprint(model):
    digest = hashlib.sha256(b'damendaemon-stage-model-v1\0')

    def metadata(value):
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)

    metadata(dict(config=model.config, context=model.max_context, quantization=model.quantization,
                  weight_format=getattr(model, 'weight_format', model.quantization), byteorder=sys.byteorder))
    for label, group in (('weights', model.weights), ('scales', model.scales)):
        metadata([label, len(group)])
        for name, tensor in sorted(group.items()):
            if tensor.device.type != 'cpu':
                raise ValueError('stage identity requires prepared CPU tensors')
            contiguous = tensor.contiguous()
            size = contiguous.numel() * contiguous.element_size()
            metadata([name, str(contiguous.dtype), list(contiguous.shape), size])
            if size:
                # Keep contiguous alive while hashlib reads its bounded CPU
                # buffer. This avoids a NumPy dependency or a full bytes copy.
                buffer = (ctypes.c_ubyte * size).from_address(contiguous.data_ptr())
                digest.update(buffer)
    return digest.digest()

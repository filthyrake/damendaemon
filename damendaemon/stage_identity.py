"""Content identity for prepared CPU weights before stage-group activation."""
import ctypes
import hashlib
import json
import sys


def chain_contract(config, context, quantizations, boundary_dtype='float32',
                  checkpoint_digest=None):
    """The shared startup contract of a (possibly mixed-format) chain (#601).

    Uniform chains pin identity through fingerprint(): every rank loads the
    same quantization, so loaded weight bytes must agree. A mixed-format
    chain cannot — each rank loads the checkpoint in its own format — so
    the shared contract pins what every rank can still derive identically
    from its own load: the checkpoint-sourced decoder config, the context,
    the full per-rank format vector (every rank knows the whole list), the
    boundary activation dtype and — when supplied — a checkpoint-derived
    digest. A disagreement aborts startup before serving, exactly like the
    uniform fingerprint gate; per-rank fingerprints stay in each rank's
    local report.

    WITHOUT `checkpoint_digest` the contract CANNOT detect a wrong
    checkpoint: two ranks holding different checkpoints with identical
    architecture (different fine-tunes of the same model) agree on
    everything above. Callers that stage checkpoints with a CRC manifest
    (the gate-4 discipline's crc32.txt) must hash their own copy's
    manifest and pass it here; identical checkpoint copies derive the
    identical digest, and a wrong or stale copy aborts startup.
    """
    if (not isinstance(quantizations, (list, tuple)) or not 2 <= len(quantizations) <= 16
            or any(not isinstance(value, str) or not value or len(value) > 32
                   for value in quantizations)
            or not isinstance(context, int) or context <= 0
            or not isinstance(boundary_dtype, str) or not boundary_dtype
            or (checkpoint_digest is not None
                and (not isinstance(checkpoint_digest, (bytes, bytearray))
                     or len(checkpoint_digest) != 32))):
        raise ValueError('chain contract requires 2..16 named formats, a context, '
                         'a boundary dtype and a 32-byte checkpoint digest or None')
    metadata = dict(config=config, context=context,
                    per_rank_quantization=[str(value) for value in quantizations],
                    boundary_activation_dtype=boundary_dtype, byteorder=sys.byteorder)
    if checkpoint_digest is not None:
        metadata['checkpoint_digest'] = bytes(checkpoint_digest).hex()
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return hashlib.sha256(b'damendaemon-chain-contract-v1\0' + encoded).digest()


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

"""Cross-rank transport-version agreement for distributed startup.

The startup digest previously covered weights/protocol/context/batch/packing
only; mixed torch/NCCL builds across ranks failed at NCCL rendezvous with
opaque socket errors (gate 4). The payload gathered here adds the transport
identity so a mismatch is rejected explicitly, naming both sides, before
any collective transport initialization depends on it.
"""
import json
import struct

MAX_PAYLOAD_BYTES = 4096


def transport_identity(torch):
    hip = getattr(torch, 'version', None) and torch.version.hip
    if hip:
        backend = 'rccl'
        runtime = hip
    elif torch.version.cuda:
        backend = 'nccl'
        runtime = torch.version.cuda
    else:
        backend = 'none'
        runtime = None
    if backend == 'nccl':
        transport = '.'.join(map(str, torch.cuda.nccl.version()))
    else:
        transport = 'bundled'
    return dict(torch=torch.__version__, backend=backend, runtime=runtime,
                transport=transport)


def session_payload(identity_table, bounds, transport, protocol_version):
    """Serialize the full startup agreement: model table, bounds, transport."""
    payload = json.dumps(dict(models=identity_table, bounds=bounds,
                              protocol=protocol_version, transport=transport),
                         sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    encoded = payload.encode()
    if not 0 < len(encoded) <= MAX_PAYLOAD_BYTES:
        raise ValueError('distributed startup payload exceeds its bounded size')
    return encoded


def compare_payloads(local_bytes, peer_bytes, local_rank=0):
    """Raise an explicit mismatch naming both sides; return the verified payload.

    Equality is on the full payload, so a model-table, bounds, protocol or
    transport difference is reported with the offending values instead of an
    opaque hash inequality or a socket failure at rendezvous.
    """
    if local_bytes == peer_bytes:
        return json.loads(local_bytes)
    local = json.loads(local_bytes)
    peer = json.loads(peer_bytes)
    peer_rank = 1 - local_rank
    for key in ('transport', 'protocol', 'bounds', 'models'):
        if local.get(key) != peer.get(key):
            raise ValueError(
                f'distributed startup mismatch on {key}: rank {local_rank} has '
                f'{json.dumps(local.get(key), sort_keys=True)} vs rank {peer_rank} has '
                f'{json.dumps(peer.get(key), sort_keys=True)}; '
                + ('align venvs/torch builds before rendezvous (gate 4 finding)'
                   if key == 'transport' else 'use identical model tables and bounds on both ranks'))
    raise ValueError('distributed startup mismatch between ranks')

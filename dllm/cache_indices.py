"""Causal index plans for full-prefill and single-token ring-cache attention.

Independent index geometry follows DeepSeek-V4-Flash inference/model.py (MIT),
revision 60d8d70770c6776ff598c94bb586a859a38244f1. No cache mutation occurs here.
"""
import torch

MAX_INDEX_ELEMENTS = 1_048_576


def _integer(value, name, minimum=1):
    if type(value) is not int or not minimum <= value < 2**31:
        raise ValueError(f'{name} must be an integer in {minimum}..2147483647')


def _options(window, ratio, batch, rows, columns):
    _integer(window, 'window')
    _integer(ratio, 'compression ratio', 0)
    _integer(batch, 'batch')
    if batch*rows*columns > MAX_INDEX_ELEMENTS:
        raise ValueError('cache index plan exceeds 1048576 elements; split query rows')


def prefill_indices(sequence_length, *, window, ratio=0, batch=1,
                    query_start=0, query_stop=None, device='cpu'):
    """Indices into [full sequence KV, completed compressed blocks].

    Query slicing bounds planning workspace without changing cache geometry.
    A compressed block becomes visible only after all ratio tokens are present.
    The caller must finish producing the visible compressed vectors before use.
    """
    _integer(sequence_length, 'sequence length')
    _integer(query_start, 'query start', 0)
    query_stop = sequence_length if query_stop is None else query_stop
    _integer(query_stop, 'query stop')
    _integer(window, 'window')
    _integer(ratio, 'compression ratio', 0)
    if not query_start < query_stop <= sequence_length:
        raise ValueError('prefill query range must be nonempty and within the full sequence')
    keys = min(window, sequence_length)
    compressed = query_stop//ratio if ratio else 0
    _options(window, ratio, batch, query_stop-query_start, keys+compressed)
    positions = torch.arange(query_start, query_stop, device=device)
    starts = (positions-window+1).clamp_min(0)
    local = starts[:, None]+torch.arange(keys, device=device)
    local = torch.where(local <= positions[:, None], local, -1)
    if compressed:
        blocks = torch.arange(compressed, device=device)
        compressed_ids = torch.where(blocks[None, :] < (positions[:, None]+1)//ratio,
                                     blocks[None, :]+sequence_length, -1)
        local = torch.cat((local, compressed_ids), dim=-1)
    return local.unsqueeze(0).expand(batch, -1, -1)


def ring_decode_indices(positions, *, window, ratio=0, device='cpu'):
    """One query per slot, using [window ring KV, compressed blocks].

    positions are scheduler-owned logical token positions, one Python integer
    per batch slot; their order may change. Each current token must be written at
    position % window before attention. Multi-token ring writes are deliberately
    not represented: they can overwrite context needed by earlier query rows.
    """
    if not isinstance(positions, (tuple, list)) or not positions:
        raise ValueError('ring decode requires a nonempty list/tuple of logical positions')
    for position in positions:
        _integer(position, 'logical position', 0)
    _integer(window, 'window')
    _integer(ratio, 'compression ratio', 0)
    compressed = (max(positions)+1)//ratio if ratio else 0
    _options(window, ratio, len(positions), 1, window+compressed)
    logical = torch.tensor(positions, dtype=torch.int64, device=device)
    starts = (logical-window+1).clamp_min(0)
    ordered = starts[:, None]+torch.arange(window, device=device)
    local = torch.where(ordered <= logical[:, None], ordered % window, -1)
    if compressed:
        blocks = torch.arange(compressed, device=device)
        compressed_ids = torch.where(blocks[None, :] < (logical[:, None]+1)//ratio,
                                     blocks[None, :]+window, -1)
        local = torch.cat((local, compressed_ids), dim=-1)
    return local.unsqueeze(1)

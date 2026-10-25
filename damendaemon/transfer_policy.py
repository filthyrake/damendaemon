"""#604 gate 3: disclosed transfer-path selection and bounded pinned staging.

Peer access is enabled only where the runtime reports it is mutually
available; every attempt produces a disclosure row rather than an
exception. Pinned staging (opt-in via DAMENDAEMON_PINNED_STAGING) stages
host payloads through a pinned buffer bounded by the caller's existing
max_transfer_bytes budget, so host round-trips stop paying the pageable
copy penalty without any unbounded host allocation.
"""
import os

import torch

ENABLE_VALUES = ('', '1', 'true', 'on')
DISABLE_VALUES = ('0', 'false', 'off')


def pinned_staging_enabled(env=None):
    """Strict tri-state: opt-in, default off, garbage raises."""
    value = os.environ.get('DAMENDAEMON_PINNED_STAGING', '0') if env is None else env
    if value in ENABLE_VALUES:
        return True
    if value in DISABLE_VALUES:
        return False
    raise ValueError('DAMENDAEMON_PINNED_STAGING must be a disable value '
                     f'{DISABLE_VALUES} or an enable value {ENABLE_VALUES}')


def peer_plan(torch_module):
    """Which ordered device pairs the runtime says can reach peer access."""
    available = bool(getattr(torch_module.cuda, 'is_available', lambda: False)())
    count = torch_module.cuda.device_count() if available else 0
    rows = []
    for source in range(count):
        for destination in range(count):
            if source == destination:
                continue
            try:
                mutual = bool(torch_module.cuda.can_device_access_peer(source, destination))
            except Exception as error:
                rows.append(dict(source=source, destination=destination,
                                 peer_available=False, error=str(error)))
                continue
            rows.append(dict(source=source, destination=destination, peer_available=mutual))
    return rows


def enable_peer_access(torch_module):
    """Enable P2P for every mutually-capable ordered pair; disclose the rest.

    Uses the CUDA runtime directly so an already-enabled pair is tolerated
    (error 701) rather than raised; a genuine failure is disclosed and the
    pair falls back to the staged driver path. No staging memory is
    allocated here.
    """
    disclosures = []
    cudart = getattr(torch_module.cuda, 'cudart', None)
    for row in peer_plan(torch_module):
        entry = dict(row)
        if row['peer_available'] and cudart is not None:
            try:
                with torch_module.cuda.device(row['source']):
                    result = cudart().cudaDeviceEnablePeerAccess(row['destination'])
            except Exception as error:
                entry.update(enabled=False, error=str(error))
            else:
                # cudaSuccess == 0; cudaErrorPeerAccessAlreadyEnabled == 701
                if result in (0, 701):
                    entry.update(enabled=True)
                else:
                    entry.update(enabled=False, error=f'cudaDeviceEnablePeerAccess -> {result}')
        else:
            entry.update(enabled=False,
                         error=row.get('error') or 'peer access unavailable per runtime report')
        disclosures.append(entry)
    return disclosures


def _pinned_empty(numel, dtype):
    """Pinned allocation, injectable so CPU-only contract runs can mock it."""
    return torch.empty(numel, dtype=dtype, pin_memory=True)


def _h2d(buffer_slice, device):
    """Upload a pinned host slice; injectable for the same reason."""
    return buffer_slice.to(device, non_blocking=False)


class PinnedStaging:
    """Round-trip staging through a pinned host buffer, bounded by budget.

    One buffer per direction is grown lazily to the largest payload seen and
    never beyond the caller's max_transfer_bytes budget. H2D copies the
    caller's tensor into pinned memory, then uploads; D2H downloads into
    pinned memory, then clones out so callers still receive an ordinary
    unpinned contiguous CPU tensor, matching the boundary contract. With
    staging disabled this degrades to the historical blocking pageable copy.
    """

    def __init__(self, device, *, max_transfer_bytes, enabled=None):
        if enabled is None:
            enabled = pinned_staging_enabled()
        self.device = device
        self.enabled = bool(enabled)
        self.budget = max_transfer_bytes
        self._up = None
        self._down = None
        self.bytes_staged = 0

    def _buffer(self, slot, numel, dtype):
        if slot is None or slot.numel() < numel or slot.dtype != dtype:
            if numel * dtype.itemsize > self.budget:
                raise ValueError('host CUDA boundary transfer payload budget exceeded')
            slot = _pinned_empty(numel, dtype=dtype)
        return slot

    def _flat(self, value):
        return value.contiguous().view(-1) if value.numel() else value.view(-1)

    def to_device(self, value):
        if not self.enabled or value.device.type != 'cpu':
            return value.to(self.device, copy=True).contiguous()
        flat = self._flat(value)
        self._up = self._buffer(self._up, flat.numel(), flat.dtype)
        self._up[:flat.numel()].copy_(flat)
        self.bytes_staged += flat.numel() * flat.element_size()
        return _h2d(self._up[:flat.numel()], self.device).view_as(value).contiguous()

    def to_host(self, value):
        if not self.enabled or value.device.type == 'cpu':
            return value.to('cpu', copy=True).contiguous()
        flat = value.contiguous().view(-1)
        self._down = self._buffer(self._down, flat.numel(), flat.dtype)
        self._down[:flat.numel()].copy_(flat, non_blocking=False)
        self.bytes_staged += flat.numel() * flat.element_size()
        return self._down[:flat.numel()].clone().view_as(value).contiguous()

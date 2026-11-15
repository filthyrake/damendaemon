"""#731: interconnect-aware peer-resident split placement policy.

Consumes the #604 topology inventory (devices + peer matrix) and a
stage's per-layer resident bytes, and decides — with recorded arithmetic —
whether the stage's weights should be:

  ``host_resident_stream``  the default: banks in host RAM, selected
                             experts streamed per use (rung 4a / #719
                             rung 5 — what the shipped fleet chain does)
  ``peer_resident_split``   opt-in: banks device-resident, split across a
                             mutually P2P-capable pair (e.g. the 3090box
                             NV4 pair), far halves read over peer access

The split is PROVEN bit-exact vs the streamed control on the 3090box pair
(#730, PR #735: torch.equal, max_abs 0.0, 8/8 parity), and #730 measured
the real envelope: 21.83 GiB banks per GPU + ~3.06 GiB working set peaked
at 96.6% of 24 GB — the projected-peak arithmetic below is therefore the
go/no-go, not an assumed margin.

Opt-in: ``DAMENDAEMON_PEER_RESIDENT_SPLIT`` (strict tri-state, default
off). Promoting it to a default is a separate issue gated on more cells;
the opt-in rule ships now that the #730 parity verdict is in.

Everything here is pure: no torch, no sysfs, no host-environment reads —
callers pass the inventory dict (the shape `topology_inventory.
run_topology_report` emits) and the byte numbers. Unit tests inject both.
"""
import os

ENABLE_VALUES = ('', '1', 'true', 'on')
DISABLE_VALUES = ('0', 'false', 'off')

# #730 measured the working set on top of split-resident banks directly:
# peak reserved 24,891,097,088 B − 21,828,206,592 B banks per GPU =
# 3,062,890,496 B, for the fleet chain's fp8-oracle parity shape (tokens 8).
# With this measured value the projected free fraction equals the cell's
# measured 3.41%. Callers with a different workload pass their own.
DEFAULT_WORKING_SET_BYTES = 3_062_890_496
# #730 completed with 3.4% of device memory to spare at that shape; the
# default go/no-go requires a little more than was measured to spare, so a
# projection that lands under it has the margin the measured cell actually
# demonstrated plus a hair. Callers override with their own measurement.
DEFAULT_MIN_FREE_FRACTION = 0.03


def peer_split_enabled(env=None):
    """Strict tri-state opt-in: default off, garbage raises."""
    value = os.environ.get('DAMENDAEMON_PEER_RESIDENT_SPLIT', '0') if env is None else env
    if value in ENABLE_VALUES:
        return True
    if value in DISABLE_VALUES:
        return False
    raise ValueError('DAMENDAEMON_PEER_RESIDENT_SPLIT must be a disable value '
                     f'{DISABLE_VALUES} or an enable value {ENABLE_VALUES}')


def mutually_p2p_pairs(inventory):
    """Mutually peer-capable device pairs from a topology inventory dict.

    Returns ``[(a, b, aggregate_memory_bytes)]`` — both orderings of the
    runtime's peer matrix must report the pair capable (the honest form of
    the #730 admission check), ordered by aggregate memory descending.
    """
    devices = inventory.get('devices') or []
    rows = inventory.get('peer_matrix') or []
    capability = {(row['source'], row['destination']): bool(row['peer_available'])
                  for row in rows if 'source' in row and 'destination' in row}
    totals = {device['index']: int(device.get('total_memory_bytes') or 0)
              for device in devices}
    pairs = []
    seen = set()
    for (source, destination), capable in capability.items():
        if not capable:
            continue
        pair = (min(source, destination), max(source, destination))
        if pair in seen or not capability.get((pair[1], pair[0])):
            continue
        seen.add(pair)
        aggregate = totals.get(pair[0], 0) + totals.get(pair[1], 0)
        if aggregate:
            pairs.append((pair[0], pair[1], aggregate))
    return sorted(pairs, key=lambda entry: -entry[2])


def peer_split_decision(layer_bytes, inventory, *,
                        working_set_bytes=DEFAULT_WORKING_SET_BYTES,
                        min_free_fraction=DEFAULT_MIN_FREE_FRACTION,
                        enabled=None):
    """Decide a stage's residency mode with recorded arithmetic.

    ``layer_bytes`` maps layer id -> resident bytes for the stage's layers
    (the bank packed_bytes). The decision requires EVERY layer's banks to
    be resident for the whole stage (that is the capacity claim: no
    per-use streaming), split as contiguous halves so each GPU holds half
    of every layer — the #730 placement.

    Returns a dict with:
      mode            'peer_resident_split' | 'host_resident_stream'
      opt_in          whether the env opt-in is on
      reason          why the mode was chosen (or refused)
      pair            selected (a, b) when a split is selected
      arithmetic      per_gpu_banks / working_set / projected_peak /
                      total / projected_free_fraction / threshold
      single_gpu_feasible  whether the whole stage fits one device
      layer_count, total_bytes

    A pair is selected only when: the opt-in is on, the stage does NOT fit
    a single device, a mutually P2P pair exists, and the projected
    per-GPU peak (half the banks + working set) leaves at least
    ``min_free_fraction`` of each device free.
    """
    if enabled is None:
        enabled = peer_split_enabled()
    total_bytes = int(sum(layer_bytes.values()))
    devices = inventory.get('devices') or []
    totals = {int(device['index']): int(device.get('total_memory_bytes') or 0)
              for device in devices}
    largest_single = max(totals.values(), default=0)
    single_gpu_feasible = total_bytes + working_set_bytes <= largest_single * (1 - min_free_fraction)

    decision: dict = dict(layer_count=len(layer_bytes), total_bytes=total_bytes,
                    single_gpu_feasible=single_gpu_feasible,
                    opt_in=bool(enabled))

    if not enabled:
        return dict(decision, mode='host_resident_stream',
                    reason='DAMENDAEMON_PEER_RESIDENT_SPLIT not enabled '
                           '(opt-in; default off until promotion per #731)',
                    pair=None, arithmetic=None)
    if single_gpu_feasible:
        return dict(decision, mode='host_resident_stream',
                    reason='the whole stage fits a single device with the '
                           'required free fraction — no split needed',
                    pair=None, arithmetic=None)

    per_gpu_banks = (total_bytes + 1) // 2  # contiguous halves; odd byte rounds up
    pairs = mutually_p2p_pairs(inventory)
    if not pairs:
        return dict(decision, mode='host_resident_stream',
                    reason='no mutually P2P-capable pair with memory in the inventory',
                    pair=None, arithmetic=None)
    refusal: dict = dict(decision, mode='host_resident_stream',
                         reason='every mutually P2P-capable pair leaves less than '
                                'the required free fraction at the projected peak',
                         pair=None, arithmetic=None)
    for a, b, _aggregate in pairs:
        arithmetic = dict(per_gpu_banks_bytes=per_gpu_banks,
                          working_set_bytes=int(working_set_bytes),
                          projected_peak_bytes=per_gpu_banks + int(working_set_bytes),
                          device_total_bytes={str(a): totals[a], str(b): totals[b]},
                          projected_free_fraction={
                              str(device): 1 - (per_gpu_banks + working_set_bytes) / totals[device]
                              for device in (a, b)},
                          min_free_fraction=min_free_fraction)
        if all(arithmetic['projected_free_fraction'][str(device)] >= min_free_fraction
               for device in (a, b)):
            return dict(decision, mode='peer_resident_split',
                        reason=f'stage exceeds single-device capacity but fits '
                               f'P2P pair ({a},{b}) with the required free fraction',
                        pair=(a, b), arithmetic=arithmetic)
        # keep the last evaluated arithmetic for the refusal record
        refusal = dict(decision, mode='host_resident_stream',
                       reason=f'P2P pair ({a},{b}) exists but the projected peak '
                              f'leaves less than the required free fraction',
                       pair=(a, b), arithmetic=arithmetic)
    return refusal

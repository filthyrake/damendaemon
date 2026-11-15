"""#731 validation: the planner picks the peer split on real topology.

Guarded cell on the 3090box dual-3090 pair. Builds the live topology
inventory (the topology_inventory shape), reads the REAL per-layer q4 bank
sizes from the parked #719/#730 bank files, and runs the #731
peer-split decision:

  - layers [58,66): must select peer_resident_split on the P2P pair —
    the range #730 proved bit-exact at the projected arithmetic.
  - a 9-layer range: must REFUSE on headroom (the #730 measured boundary).
  - opt-in unset: must stay host_resident_stream (default off until a
    promotion issue).

Pre-declared gates (#731):
1. live_inventory: exactly the two 3090s, mutual peer access both ways.
2. plan_picks_split: decision mode == peer_resident_split, pair selected,
   projected free fraction >= the policy threshold on both GPUs.
3. arithmetic_matches_730: projected peak == #730's measured peak reserved
   (24,891,097,088 B) exactly — the projection reproduces the measurement.
4. nine_layers_refused: synthetic 9th layer (layer-58 size as proxy;
   layer-66's bank lives on homer and is unreadable from here — disclosed)
   refuses with the headroom reason and records the refused arithmetic.
5. opt_in_off_by_default: env unset -> host_resident_stream, opt_in False.
No throughput/SLO claim; placement-policy validation only.
"""
if not __debug__:
    raise RuntimeError('verify_peer_split_planner_3090box asserts gates; '
                       'run without -O')

import argparse
import json
import time
from pathlib import Path

import torch

from damendaemon.peer_split_policy import peer_split_decision
from damendaemon.topology_inventory import peer_matrix
from scripts.experiments.split_gguf import save

LAYER_START, LAYER_STOP = 58, 66
# #730 split-3 measured peak reserved on the executing GPU
MEASURED_PEAK = 24_891_097_088
MEASURED_WORKING_SET = 3_062_890_496


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--banks-dir', default='/home/damen/damendaemon-g719r5/banks')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate bank files exist; no inventory/GPU work')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new report path required')

    layer_bytes = {}
    for layer in range(LAYER_START, LAYER_STOP):
        path = Path(args.banks_dir) / f'layer{layer}.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        layer_bytes[layer] = None  # filled by the live read below
    if args.dry_run:
        print('DRYRUN_OK', len(layer_bytes), 'bank files present')
        return

    report: dict = dict(mechanism='peer_split_planner_validation',
                        layer=[LAYER_START, LAYER_STOP], torch=torch.__version__,
                        gates=[])
    save(args.output, report)

    # ---- live inventory (topology_inventory shape)
    count = torch.cuda.device_count()
    devices = [dict(index=i, total_memory_bytes=torch.cuda.get_device_properties(i).total_memory)
               for i in range(count)]
    inventory = dict(devices=devices, peer_matrix=peer_matrix(torch, count))
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    report['device'] = names
    mutual = (count == 2 and all('3090' in n for n in names)
              and all(row['peer_available'] for row in inventory['peer_matrix']))
    report['gates'].append(dict(gate='live_inventory', passed=bool(mutual),
                                devices=names, peer_matrix=inventory['peer_matrix']))
    save(args.output, report)

    # ---- real bank sizes (live metadata read; tensors freed per layer)
    began = time.monotonic()
    for layer in range(LAYER_START, LAYER_STOP):
        payload = torch.load(Path(args.banks_dir) / f'layer{layer}.pt',
                             map_location='cpu', weights_only=True)
        layer_bytes[layer] = int(payload['packed_bytes'])
        del payload
    report['bank_read_seconds'] = round(time.monotonic() - began, 1)

    # ---- gate 2: the 8-layer range selects the split
    decision: dict = peer_split_decision(layer_bytes, inventory,
                                   working_set_bytes=MEASURED_WORKING_SET,
                                   enabled=True)
    picks = decision['mode'] == 'peer_resident_split' and decision['pair'] == (0, 1)
    report['gates'].append(dict(gate='plan_picks_split', passed=bool(picks),
                                decision=decision))
    save(args.output, report)

    # ---- gate 3: the projection reproduces #730's measured peak exactly
    projected = decision['arithmetic']['projected_peak_bytes']
    report['gates'].append(dict(gate='arithmetic_matches_730',
                                passed=bool(projected == MEASURED_PEAK),
                                projected_peak_bytes=projected,
                                measured_peak_bytes=MEASURED_PEAK))
    save(args.output, report)

    # ---- gate 4: the 9-layer range refuses on headroom (disclosed proxy)
    nine = dict(layer_bytes)
    nine[66] = layer_bytes[LAYER_START]  # same-architecture proxy; homer bank unreadable
    refusal: dict = peer_split_decision(nine, inventory,
                                   working_set_bytes=MEASURED_WORKING_SET,
                                   enabled=True)
    refused = (refusal['mode'] == 'host_resident_stream'
               and 'projected peak' in refusal['reason'])
    report['gates'].append(dict(
        gate='nine_layers_refused', passed=bool(refused),
        note='layer-66 size proxied by layer-58 (same architecture; its bank '
             'is parked on homer and not readable from 3090box)',
        decision=refusal))
    save(args.output, report)

    # ---- gate 5: opt-in stays off by default
    default = peer_split_decision(layer_bytes, inventory,
                                   working_set_bytes=MEASURED_WORKING_SET)
    report['gates'].append(dict(gate='opt_in_off_by_default',
                                passed=bool(default['mode'] == 'host_resident_stream'
                                            and not default['opt_in']),
                                reason=default['reason']))

    report['passed'] = all(gate['passed'] for gate in report['gates'])
    save(args.output, report)
    print('PLANNER_VALIDATION_DONE', json.dumps(dict(passed=report['passed'])))


if __name__ == '__main__':
    main()

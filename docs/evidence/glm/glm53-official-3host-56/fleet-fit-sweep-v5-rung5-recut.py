"""#56 rung 5: boundary-legal range re-cut for the 3-host topology cell.

The fleet-fit v4 greedy plan (weights-only accounting) cut
[[0,66),[66,77),[77,78)] by bytes alone. Two defects, both structural:

1. ``validate_range`` requires every stage to START on a 'full' indexer
   layer (full layers: 0,1,2, then every 4th from 6 through 74). Layer 77
   is a 'shared' layer, so [77,78) is refused at range validation - after
   the whole-snapshot identity hash.
2. Even the legal variant [0,66) needs 585.99 GiB against ~579 GiB
   actually allocatable on the expanded labnode3 (589 GiB visible minus
   its resident baseline) - an honest overcommit.

This sweep enumerates ALL boundary-legal 3-stage cuts (each start on a
full layer, final stage absorbing the whole tail group [74,78) plus
norm/head) using the same contract arithmetic as fleet-fit v4
(``glm_checkpoint.tensor_specs``, fp8 retained 1B / floating retained
fp32 / per-tensor source-conversion overlap), and reports the one cut
with real headroom on every host. 2026-11-08.
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from damendaemon.glm_checkpoint import tensor_specs  # noqa: E402

fixture = json.load(open(ROOT / 'tests/fixtures/glm53_contract.json'))
config = fixture['config']
specs = tensor_specs(config)
layers = config['num_hidden_layers']
full = [i for i, kind in enumerate(config['indexer_types']) if kind == 'full']

ITEM = {'fp8': 1, 'floating': 4, 'scale': 4}
SRC = {'floating': 2, 'fp8': 1, 'scale': 4}
retained = [0] * (layers + 2)
overlap = [0] * (layers + 2)
for name, spec in specs.items():
    e = math.prod(spec.shape)
    if name == 'model.embed_tokens.weight':
        idx = 0
    elif name.startswith('model.layers.'):
        idx = 1 + int(name.split('.')[2])
    else:
        idx = layers + 1
    retained[idx] += e * ITEM[spec.storage]
    overlap[idx] = max(overlap[idx], e * SRC[spec.storage])
GIB = 2 ** 30
MIB = 2 ** 20


def need(a, b):
    n = sum(retained[a + 1:b + 1])
    o = max(overlap[a + 1:b + 1]) if b > a else 0
    if a == 0:
        n += retained[0]
        o = max(o, overlap[0])
    if b == layers:
        n += retained[layers + 1]
        o = max(o, overlap[layers + 1])
    return n, o


hosts = [('labnode3', 579), ('rocky10-desktop', 111), ('3090box', 122)]  # allocatable GiB, measured 2026-11-08
candidates = []
chosen = None
for b1 in full:
    if not 0 < b1 <= 74:
        continue
    for b2 in full:
        # final stage [b2,78) must start on a full layer and absorb the tail group [74,78)
        if not b1 < b2 <= 74:
            continue
        ranges = [(0, b1), (b1, b2), (b2, 78)]
        needs = [need(a, b) for a, b in ranges]
        fits = all(n + o <= cap * GIB for (n, o), (_, cap) in zip(needs, hosts))
        margin = min((cap * GIB - (n + o)) / GIB for (n, o), (_, cap) in zip(needs, hosts))
        row = {'ranges': [list(r) for r in ranges],
               'stage_need_gib': [round((n + o) / GIB, 2) for n, o in needs],
               'min_host_margin_gib': round(margin, 2), 'fits': fits}
        candidates.append(row)
        if fits and (chosen is None or margin > chosen[1]):
            chosen = (ranges, margin)

if chosen is None:
    raise SystemExit('no boundary-legal 3-stage cut fits the measured hosts')
ranges, margin = chosen
needs = [need(a, b) for a, b in ranges]
report = {
    'scope': 'glm53_official_rung5_range_recut',
    'date': '2026-11-08',
    'official_revision': fixture['revision'],
    'full_indexer_layers': full,
    'hosts_allocatable_gib': {name: cap for name, cap in hosts},
    'recut_reason': [
        'v4 greedy [0,66),[66,77),[77,78) starts a stage on shared layer 77 (validate_range refuses)',
        'legal variant [0,66) needs 586.0 GiB vs ~579 GiB allocatable on labnode3',
    ],
    'chosen_plan': {
        'ranges': [list(r) for r in ranges],
        'hosts': ['labnode3', 'rocky10-desktop', '3090box'],
        'stage_retained_gib': [round(n / GIB, 2) for n, o in needs],
        'stage_overlap_gib': [round(o / GIB, 2) for n, o in needs],
        'stage_peak_gib': [round((n + o) / GIB, 2) for n, o in needs],
        'weight_mib_flags': [math.ceil((n + o) / MIB) + 1 for n, o in needs],
        'min_host_margin_gib': round(margin, 2),
    },
    'candidates': candidates,
}
out = Path(__file__).resolve().parent / 'fleet-fit-sweep-v5-rung5-recut.json'
json.dump(report, open(out, 'w'), indent=1, sort_keys=True)
print('chosen:', ranges, 'min margin %.1f GiB' % margin)
for (n, o), (name, _) in zip(needs, hosts):
    print('  %-16s retained %7.2f GiB overlap %5.2f GiB peak %7.2f GiB weight-mib %d'
          % (name, n / GIB, o / GIB, (n + o) / GIB, math.ceil((n + o) / MIB) + 1))

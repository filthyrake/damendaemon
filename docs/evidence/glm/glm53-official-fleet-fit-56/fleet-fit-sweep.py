"""#56 fleet fit sweep for official GLM-5.3 — contract-derived, no payload reads.

Per-stage required bytes use the planner's own retention arithmetic
(fp8 storage 1B/item, floating retained as FP32 4B/item, scale 4B/item,
plus the largest single-tensor source conversion overlap at a boundary).
Cache and workspace allowances are deliberately ZERO: the no-fit verdict is
weights-alone, the most generous possible assumption.
"""
import json, math, sys
sys.path.insert(0, '/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56')
from damendaemon.glm_checkpoint import tensor_specs

fixture = json.load(open('/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56/tests/fixtures/glm53_contract.json'))
config = fixture['config']
specs = tensor_specs(config)
layers = config['num_hidden_layers']

ITEM = {'fp8': 1, 'floating': 4, 'scale': 4}
SOURCE = {'floating': 2, 'fp8': 1, 'scale': 4}  # on-disk dtype bytes (bf16/fp8/f32)

retained = [0.0] * (layers + 2)   # [0]=embed, [1..layers]=layers, [layers+1]=norm+head
overlap = [0] * (layers + 2)
for name, spec in specs.items():
    elements = math.prod(spec.shape)
    if name == 'model.embed_tokens.weight':
        idx = 0
    elif name.startswith('model.layers.'):
        idx = 1 + int(name.split('.')[2])
    else:
        idx = layers + 1
    retained[idx] += elements * ITEM[spec.storage]
    overlap[idx] = max(overlap[idx], elements * SOURCE[spec.storage])

retained = [int(x) for x in retained]
overlap = [int(x) for x in overlap]
total_retained = sum(retained)
GIB = 2**30

def stage_bytes(ranges):
    out = []
    for a, b in ranges:  # [a,b) global layers
        need = sum(retained[a+1:b+1])
        conv = max([overlap[a+1:b+1][0] if b > a else 0] + [overlap[i] for i in range(a+1, b+1)])
        if a == 0: need += retained[0]; conv = max(conv, overlap[0])
        if b == layers: need += retained[layers+1]; conv = max(conv, overlap[layers+1])
        out.append(need + conv)
    return out

def balanced(k):
    base, rem = divmod(layers, k)
    ranges, start = [], 0
    for i in range(k):
        n = base + (1 if i < rem else 0)
        ranges.append((start, start+n)); start += n
    return ranges

fleet_hosts = ['labnode3 (118 GiB)', 'homer (122 GiB)', 'rocky10-desktop (124 GiB)', '3090box (125 GiB)']
budget = 110 * GIB  # uniform conservative per-host budget (same convention as #703/#709 sweeps)

report = {
    'scope': 'glm53_official_fleet_fit_weights_alone',
    'family': 'glm', 'official_revision': fixture['revision'],
    'checkpoint_total_bytes': fixture['official_checkpoint_total_bytes'],
    'main_model_storage_bytes': fixture['main_model_storage_bytes'],
    'retained_converted_bytes': total_retained,
    'retained_converted_gib': round(total_retained / GIB, 2),
    'weights_only_zero_cache': True,
    'fleet_hosts': fleet_hosts,
    'fleet_host_budget_bytes': budget,
    'fleet_total_bytes': budget * len(fleet_hosts),
    'sweep': [],
}
k_min = None
for k in range(2, 17):
    ranges = balanced(k)
    needs = stage_bytes(ranges)
    entry = {'stages': k, 'ranges': [list(r) for r in ranges],
             'max_stage_bytes': max(needs),
             'max_stage_gib': round(max(needs) / GIB, 2),
             'fits_110gib_hosts': max(needs) <= budget}
    report['sweep'].append(entry)
    if k_min is None and max(needs) <= budget:
        k_min = k
report['min_stages_for_110gib_hosts'] = k_min
report['fleet_capable_hosts'] = len(fleet_hosts)
report['verdict'] = ('no-fit: fleet-RAM-gated' if (k_min or 99) > len(fleet_hosts)
                     else 'fit possible with %d hosts' % k_min)
out = '/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56/docs/evidence/glm/glm53-official-fleet-fit-56/fleet-fit-sweep.json'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(report, open(out, 'w'), indent=1, sort_keys=True)
print('retained converted:', round(total_retained/GIB,2), 'GiB (weights alone, no cache)')
print('fleet total (4x110GiB):', round(budget*4/GIB,1), 'GiB')
for e in report['sweep']:
    print(f"k={e['stages']:2d}  max stage {e['max_stage_gib']:8.2f} GiB  fits={e['fits_110gib_hosts']}")
print('min k for 110 GiB hosts:', k_min, '| fleet has', len(fleet_hosts), 'capable hosts')
print('verdict:', report['verdict'])

"""#56 fleet-fit sweep v3: labnode3 IS archimedes VM 109 — expand its pool.

Correct topology: archimedes 1007 GiB total hosts labnode3 (currently 118 GiB)
+ other guests (323 GiB running). Expanding labnode3's allocation (Proxmox
config + VM restart) is the keystone, not a separate 'archimedes-vm'.
Physical stage hosts: rocky10-desktop 124, 3090box 125, homer 122 GiB.
"""
import json, math, sys
sys.path.insert(0, '/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56')
from damendaemon.glm_checkpoint import tensor_specs

fixture = json.load(open('/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56/tests/fixtures/glm53_contract.json'))
specs = tensor_specs(fixture['config'])
layers = fixture['config']['num_hidden_layers']
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

GIB = 2**30
total = sum(retained)


def need(a, b):  # contiguous global-layer range [a,b)
    n = sum(retained[a + 1:b + 1])
    o = max(overlap[a + 1:b + 1]) if b > a else 0
    if a == 0:
        n += retained[0]
        o = max(o, overlap[0])
    if b == layers:
        n += retained[layers + 1]
        o = max(o, overlap[layers + 1])
    return n + o


# archimedes: 1007 GiB total, other running guests hold 323 GiB (441 - 118).
# labnode3 expansion ceilings: 683 GiB without touching other guests;
# ~740 GiB trimming/stopping one idle guest. Assume 680 conservative.
fleet = [
    ('labnode3-expanded', 680),   # archimedes VM 109, pool expanded (restart required)
    ('rocky10-desktop', 110),
    ('3090box', 110),
    ('homer', 110),
]

plan = []
start = 0
for name, cap in fleet:
    if start >= layers:
        break
    end = start
    while end < layers and need(start, end + 1) <= cap * GIB:
        end += 1
    if end == start:
        continue
    plan.append((name, start, end, need(start, end)))
    start = end
covered = start >= layers

single = need(0, layers)
report = {
    'scope': 'glm53_official_fleet_fit_v3_labnode3_expansion',
    'family': 'glm',
    'official_revision': fixture['revision'],
    'checkpoint_total_bytes': fixture['official_checkpoint_total_bytes'],
    'main_model_storage_bytes': fixture['main_model_storage_bytes'],
    'retained_converted_bytes': total,
    'retained_converted_gib': round(total / GIB, 2),
    'weights_only_zero_cache': True,
    'topology_note': 'labnode3 = archimedes VM 109 (A30+A2 passed through); archimedes 1007 GiB total, other guests 323 GiB; expansion is a VM config change + restart',
    'fleet_inventory': {
        'archimedes': {'ram_total_gib': 1007, 'other_guests_gib': 323, 'labnode3_current_gib': 118},
        'physical_hosts_gib': {'rocky10-desktop': 124, '3090box': 125, 'homer': 122},
        'dedicated_vram_gb': {'A6000': 46, '3090': 24, 'A30': 24, 'A2': 15, 'total': 109},
    },
    'greedy_plan': [{'host': n, 'layers': [a, b], 'gib': round(v / GIB, 2)} for n, a, b, v in plan],
    'covered_all_layers': covered,
    'single_labnode3_expanded_gib': round(single / GIB, 2),
    'single_host_max_without_touching_guests_gib': 683,
    'verdict': 'feasible' if covered else 'infeasible',
}
out = '/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56/docs/evidence/glm/glm53-official-fleet-fit-56/fleet-fit-sweep-v3.json'
json.dump(report, open(out, 'w'), indent=1, sort_keys=True)
print(f"model: {total/GIB:.2f} GiB converted weights-alone")
print(f"covered all {layers} layers: {covered} | hosts used: {len(plan)}")
for n, a, b, v in plan:
    print(f"  {n:20s} [{a:2d},{b:2d})  {v/GIB:7.2f} GiB")
print(f"single expanded labnode3 (whole model): {single/GIB:.1f} GiB — max 683 without touching other guests, ~740+ trimming one")
print("verdict:", report['verdict'])

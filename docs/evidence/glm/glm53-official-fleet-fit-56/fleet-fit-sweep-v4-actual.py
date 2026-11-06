"""#56 fleet-fit sweep v4: ACTUAL post-expansion state.

2026-11-06: labnode3 (archimedes VM 109) expanded 118 -> 600 GiB (614400 MiB).
680 GiB was attempted and backed out: other guests actually hold ~323 GiB and
host headroom needs ~50-100 GiB, so QEMU wedged ~100 GiB short; hard-killed
and resized to 600 GiB. Host free with VM up: ~129 GiB. labnode3 sees 589 GiB.
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

def need(a, b):
    n = sum(retained[a + 1:b + 1])
    o = max(overlap[a + 1:b + 1]) if b > a else 0
    if a == 0:
        n += retained[0]
        o = max(o, overlap[0])
    if b == layers:
        n += retained[layers + 1]
        o = max(o, overlap[layers + 1])
    return n + o

fleet = [('labnode3-expanded-600g', 600), ('rocky10-desktop', 110), ('homer', 110), ('3090box', 110)]
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

report = {
    'scope': 'glm53_official_fleet_fit_v4_actual',
    'official_revision': fixture['revision'],
    'retained_converted_gib': round(total / GIB, 2),
    'weights_only_zero_cache': True,
    'actual_state': {
        'labnode3_gib_visible': 589, 'vm_config_mib': 614400,
        'archimedes_host_free_gib': 129,
        'note_680g_failure': '680 GiB attempt wedged QEMU ~100 GiB short (other guests hold ~323 GiB actual + host headroom); killed, resized 600 GiB',
    },
    'greedy_plan': [{'host': n, 'layers': [a, b], 'gib': round(v / GIB, 2)} for n, a, b, v in plan],
    'covered_all_layers': covered,
    'verdict': 'feasible' if covered else 'infeasible',
}
out = '/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56/docs/evidence/glm/glm53-official-fleet-fit-56/fleet-fit-sweep-v4-actual.json'
json.dump(report, open(out, 'w'), indent=1, sort_keys=True)
print(f"model: {total/GIB:.2f} GiB | covered: {covered}")
for n, a, b, v in plan:
    print(f"  {n:24s} [{a:2d},{b:2d})  {v/GIB:7.2f} GiB")
print('verdict:', report['verdict'])

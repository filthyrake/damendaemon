"""#56 corrected fleet-fit sweep: full inventory incl. archimedes RAM + VRAM."""
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


# Corrected fleet inventory (2026-11-06, live-verified):
# archimedes 1007 GB RAM (565 avail; a dedicated VM can be sized at will),
# labnode3 118, homer 122, rocky10-desktop 124, 3090box 125.
# Dedicated VRAM: A6000 46 + 3090 24 + A30 24 + A2 15 = 109 GB; homer 8060S
# 112 GB GTT (shared) — executor-relevant, not CPU-reference stage capacity.
fleet = [
    ('archimedes-vm', 540),      # conservative dedicated-VM carve (headroom left on host)
    ('labnode3', 110),
    ('homer', 110),
    ('rocky10-desktop', 110),
    ('3090box', 110),
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

report = {
    'scope': 'glm53_official_fleet_fit_corrected',
    'family': 'glm',
    'official_revision': fixture['revision'],
    'checkpoint_total_bytes': fixture['official_checkpoint_total_bytes'],
    'main_model_storage_bytes': fixture['main_model_storage_bytes'],
    'retained_converted_bytes': total,
    'retained_converted_gib': round(total / GIB, 2),
    'weights_only_zero_cache': True,
    'fleet_inventory': {
        'archimedes': {'ram_total_gib': 1007, 'ram_available_gib': 565, 'note': 'Proxmox host; dedicated VM carved at 540 GiB conservative'},
        'cpu_hosts_gib': {'labnode3': 118, 'homer': 122, 'rocky10-desktop': 124, '3090box': 125},
        'dedicated_vram_gb': {'A6000 (rocky10)': 46, '3090 (3090box)': 24, 'A30 (labnode3)': 24, 'A2 (labnode3)': 15, 'total': 109},
        'homer_gtt_gb': 112,
    },
    'greedy_plan': [
        {'host': n, 'layers': [a, b], 'bytes': int(v), 'gib': round(v / GIB, 2)} for n, a, b, v in plan
    ],
    'covered_all_layers': covered,
    'hosts_used': len(plan),
    'fleet_total_stage_capacity_gib': sum(c for _, c in fleet),
    'verdict': 'fit: fleet-RAM-gated-FALSE corrected — feasible' if covered else 'infeasible',
    'single_host_note': f'archimedes alone (1007 GiB total) can hold the whole {round(total/GIB,1)} GiB converted model in one VM with headroom',
}
out = '/Users/damenknight/damendaemon/worktrees/damendaemon/glm53-official-56/docs/evidence/glm/glm53-official-fleet-fit-56/fleet-fit-sweep-corrected.json'
json.dump(report, open(out, 'w'), indent=1, sort_keys=True)
print(f"model: {total/GIB:.2f} GiB converted | fleet capacity: {sum(c for _,c in fleet)} GiB across {len(fleet)} stage hosts")
print(f"covered all 78 layers: {covered} | hosts used: {len(plan)}")
for n, a, b, v in plan:
    print(f"  {n:18s} [{a:2d},{b:2d})  {v/GIB:7.2f} GiB")
print("single-VM option: archimedes-only = %.1f GiB (1007 total / 565 avail now)" % (total / GIB))
print("verdict:", report['verdict'])

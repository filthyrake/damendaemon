"""#601 cell verdict: mixed pass vs all-Q4 reference on identical boundaries.

Reads the two runner passes' captures (same split, same trace) and emits
the comparison record. The two passes feed the SAME token stream only
until the greedy streams first diverge, so every cross-pass comparison is
scoped to the input-identical rounds:

- boundary transport bit-exactness at both inter-rank boundaries, for
  every round whose fed tokens agree (up to and including the divergence
  round — its inputs are still identical, only its output argmax flips);
- the mixed head's logit divergence on those identical inputs
  (max_abs / rms_rel, exactly as measured);
- greedy token agreement, with the divergence index recorded as it lands.

Rounds after the first greedy divergence fed different tokens on the two
passes, so their boundary/logit values are excluded by construction, not
compared. Run where /tmp/dd601-{reference,mixed}.json and
/tmp/dd601-captures/ are present.
"""
if not __debug__:
    raise RuntimeError('gate4_mixed_compare asserts qualification evidence; '
                       'run Python without -O or -OO.')

import json
import math
import sys
from pathlib import Path

import torch

CAPTURES = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/dd601-captures')
reference = json.loads(Path('/tmp/dd601-reference.json').read_text())
mixed = json.loads(Path('/tmp/dd601-mixed.json').read_text())


def digests(record):
    return [entry['sha256'] for entry in record['boundary_digests']]


def load(name):
    return json.loads((CAPTURES / name).read_text())


g_ref, g_mix = reference['greedy'], mixed['greedy']
divergence = next((i for i, (a, b) in enumerate(zip(g_ref, g_mix)) if a != b),
                  len(g_ref))
comparable_rounds = divergence + 1  # rounds 0..divergence share fed tokens

verdict = dict(passed=False,
               formats=mixed['rank_formats'], bounds=mixed['bounds'],
               planned_mixed_split=mixed['planned_mixed_split'],
               greedy_divergence_index=divergence,
               comparable_rounds=comparable_rounds)

# 1. Boundary transport: for every input-identical round, both inter-rank
#    boundary streams must be bit-exact (ranks 0/1 execute the identical
#    qualified Q4 path in both passes, so identical inputs must produce
#    identical boundary activations).
ref_rank1, ref_rank2 = load('reference-rank1.json'), load('reference-rank2.json')
mix_rank1, mix_rank2 = load('mixed-rank1.json'), load('mixed-rank2.json')
for name, ref_record, mix_record in (('rank0_to_1', ref_rank1, mix_rank1),
                                     ('rank1_to_2', ref_rank2, mix_rank2)):
    ref_stream = digests(ref_record)[:comparable_rounds]
    mix_stream = digests(mix_record)[:comparable_rounds]
    assert ref_stream == mix_stream, \
        f'{name} boundary diverged on input-identical rounds'
    verdict[f'{name}_bit_exact'] = dict(rounds=comparable_rounds, bit_exact=True)
# Rounds after the divergence fed different tokens; their boundary values
# differ by construction and are excluded.
verdict['boundary_bit_exact'] = dict(
    rank0_to_1=verdict['rank0_to_1_bit_exact'],
    rank1_to_2=verdict['rank1_to_2_bit_exact'],
    post_divergence_rounds_excluded=len(g_ref) - comparable_rounds)

# 2. Head logits divergence on identical boundary inputs (rank 2 only),
#    for the comparable rounds.
ref_logits = torch.load(CAPTURES / 'reference-rank2-logits.pt', weights_only=True)
mix_logits = torch.load(CAPTURES / 'mixed-rank2-logits.pt', weights_only=True)
assert len(ref_logits) == len(mix_logits), 'logit round counts differ'
deltas = []
for label in range(comparable_rounds):
    reference_row, mixed_row = ref_logits[label], mix_logits[label]
    difference = (mixed_row - reference_row)
    max_abs = float(difference.abs().max())
    rms_rel = float(math.sqrt((difference * difference).sum()
                              / (reference_row * reference_row).clamp_min(1e-30).sum()))
    deltas.append(dict(round=label, max_abs=max_abs, rms_rel=rms_rel,
                       argmax_reference=int(reference_row.argmax()),
                       argmax_mixed=int(mixed_row.argmax())))
verdict['head_logit_divergence'] = deltas
verdict['head_argmax_agreement'] = sum(d['argmax_reference'] == d['argmax_mixed']
                                       for d in deltas)
verdict['head_max_abs'] = max(d['max_abs'] for d in deltas)
verdict['head_rms_rel_max'] = max(d['rms_rel'] for d in deltas)

# 3. Greedy agreement, recorded as it lands.
verdict['greedy_reference'] = g_ref
verdict['greedy_mixed'] = g_mix
verdict['greedy_agreement'] = sum(a == b for a, b in zip(g_ref, g_mix))
verdict['greedy_lengths'] = dict(reference=len(g_ref), mixed=len(g_mix))

# 4. The captured head argmax trail must equal the coordinator's greedy.
assert [d['argmax_mixed'] for d in deltas] == g_mix[:comparable_rounds], \
    'mixed head argmax trail disagrees with the coordinator greedy'
assert [d['argmax_reference'] for d in deltas] == g_ref[:comparable_rounds], \
    'reference head argmax trail disagrees with the coordinator greedy'

verdict['passed'] = True
out = Path('/tmp/dd601-comparison.json')
out.write_text(json.dumps(verdict, indent=2))
print(json.dumps({key: verdict[key] for key in (
    'passed', 'greedy_divergence_index', 'greedy_agreement',
    'boundary_bit_exact', 'head_argmax_agreement', 'head_max_abs',
    'head_rms_rel_max', 'comparable_rounds')}, indent=2))
print(f'wrote {out}', flush=True)

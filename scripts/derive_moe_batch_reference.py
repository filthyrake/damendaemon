"""Regroup trusted independent oracle rows without recomputing reference logits.

Only standalone replacement prefills and their original teacher-forced decode
continuations are eligible. Regrouping changes request slots, never prompts,
continuation tokens, or reference targets. Run the derived trace through the
qualified sequential runtime before using it to qualify a packed implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', help='trusted locally generated independent trace')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    source = torch.load(args.trace, map_location='cpu', weights_only=True)
    sequences = {}
    for index, event in enumerate(source['trace']):
        if event['kind'] != 'prefill':
            continue
        for row, (slot, prompt) in enumerate(zip(event['slots'], event['prompts'])):
            sequence = dict(prompt=prompt, target=event['targets'][row], continuation=[],
                            source_event=index, source_row=row)
            for following in source['trace'][index+1:]:
                if slot not in following['slots']:
                    continue
                if following['kind'] == 'prefill':
                    break
                if following['kind'] != 'decode':
                    raise ValueError('unsupported oracle event')
                selected = following['slots'].index(slot)
                sequence['continuation'].append((following['tokens'][selected], following['targets'][selected]))
            sequences.setdefault(len(prompt), sequence)
    cases = [([31, 65], [3, 1]), ([65, 31], [2, 0]), ([31, 31], [1, 3]),
             ([31]*4, [3, 1, 0, 2]), ([31, 31, 65], [2, 0, 3]), ([65, 65], [1, 0])]
    events, reports = [], []
    for lengths, slots in cases:
        chosen = [sequences[length] for length in lengths]
        steps = min(len(s['continuation']) for s in chosen)
        if steps < 4:
            raise ValueError('source oracle lacks enough independent decode steps')
        events.append(dict(kind='prefill', slots=slots, prompts=[s['prompt'] for s in chosen],
                           targets=[s['target'] for s in chosen]))
        for step in range(steps):
            order = list(range(len(slots)))
            if step % 2 == 0:
                order.reverse()
            events.append(dict(kind='decode', slots=[slots[i] for i in order],
                               tokens=[chosen[i]['continuation'][step][0] for i in order],
                               targets=[chosen[i]['continuation'][step][1] for i in order]))
        reports.append(dict(lengths=lengths, slots=slots, decode_steps=steps,
                            source_rows=[dict(event=s['source_event'], row=s['source_row']) for s in chosen]))
    provenance = dict(source_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
                      method='regroup unchanged independent per-request prompts, teacher-forced tokens and targets', cases=reports)
    torch.save({**source, 'trace': events, 'derived_provenance': provenance}, args.output)
    provenance['output_sha256'] = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    provenance['checks'] = sum(len(event['slots']) for event in events)
    Path(args.report).write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()

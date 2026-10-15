"""Summarize completed fixed ABBA phases without hiding failed cells or text drift."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from dllm.bench import distribution


def _positive_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_round(row, concurrency, length, output_tokens):
    requests = row.get('requests')
    if not isinstance(requests, list) or len(requests) != concurrency:
        raise ValueError('each measured round requires its declared request count')
    if not _positive_seconds(row.get('seconds')):
        raise ValueError('measured round requires finite positive duration')
    for request in requests:
        if not isinstance(request, dict) or request.get('passed') is not True:
            raise ValueError('failed or invalid request in measured cell')
        duration, first = request.get('seconds'), request.get('first_output_seconds')
        if (not _positive_seconds(duration) or not _positive_seconds(first)
                or not first <= duration <= row['seconds']):
            raise ValueError('inconsistent or invalid measured request timing')
        usage = request.get('usage')
        if not isinstance(usage, dict):
            raise ValueError('measured request requires exact token usage')
        for key, expected in (('prompt_tokens', length), ('completion_tokens', output_tokens)):
            if type(usage.get(key)) is not int or usage[key] != expected:
                raise ValueError('measured token usage differs from declared workload')
        if not isinstance(request.get('text'), str) or not request['text']:
            raise ValueError('measured request requires observed output text')


def summarize(directory):
    reports = {phase: json.loads((directory/f'{phase}.json').read_text())
               for phase in ('A1', 'B1', 'B2', 'A2')}
    if not all(r.get('passed') is True for r in reports.values()):
        raise ValueError('cannot summarize incomplete or failed workload phases')
    first = reports['A1']
    output_tokens = first.get('output_tokens')
    if type(output_tokens) is not int or output_tokens <= 0:
        raise ValueError('output_tokens must be a positive integer')
    candidate = reports['B1']['manifest'].get('variant') == 'hip-q8-small-batch'
    comparator = 'dllm-candidate' if candidate else 'llama.cpp'
    for phase, report in reports.items():
        if type(report.get('output_tokens')) is not int:
            raise ValueError('output_tokens must be a positive integer')
        for key in ('prompts', 'context', 'output_tokens', 'tokenizer_sha256', 'eos_policy'):
            if report[key] != first[key]:
                raise ValueError(f'{phase}: mismatched {key}')
        for key in ('model_revision', 'hardware', 'eos_ids'):
            if report['manifest'][key] != first['manifest'][key]:
                raise ValueError(f'{phase}: mismatched {key}')
        expected = 'dllm' if candidate or phase.startswith('A') else 'llama.cpp'
        if report['engine'] != expected or report['phase'] != phase:
            raise ValueError('incorrect ABBA engine order')
        if candidate and report['manifest'].get('variant') != ('ordinary' if phase.startswith('A') else 'hip-q8-small-batch'):
            raise ValueError('incorrect ABBA variant order')
        if candidate:
            for key in ('engine_revision', 'precision', 'candidate_source_sha256'):
                if not report['manifest'].get(key) or report['manifest'][key] != first['manifest'].get(key):
                    raise ValueError(f'{phase}: mismatched candidate control {key}')
    for left, right in (('A1', 'A2'), ('B1', 'B2')):
        if reports[left]['manifest'] != reports[right]['manifest']:
            raise ValueError('engine manifest changed between paired phases')
    rows = []
    for length in (128, 512, 2048):
        for concurrency in (1, 4):
            cell = dict(prompt_tokens=length, concurrency=concurrency, engines={})
            text_sets = {}
            for engine, phases in (('dllm', ('A1', 'A2')), (comparator, ('B1', 'B2'))):
                rounds = []
                for phase in phases:
                    selected = [r for r in reports[phase]['rounds'] if not r['warmup'] and
                                r['prompt_tokens'] == length and r['concurrency'] == concurrency]
                    if len(selected) != 10:
                        raise ValueError('each phase/cell requires ten measured rounds')
                    for row in selected:
                        _validate_round(row, concurrency, length, output_tokens)
                    rounds.extend(selected)
                requests = [q for r in rounds for q in r['requests']]
                if len(requests) != 20*concurrency or not all(q['passed'] for q in requests):
                    raise ValueError('failed or missing request in measured cell')
                text_sets[engine] = [{hashlib.sha256(r['requests'][i]['text'].encode()).hexdigest()
                                      for r in rounds} for i in range(concurrency)]
                cell['engines'][engine] = dict(
                    first_output_seconds=distribution([q['first_output_seconds'] for q in requests]),
                    request_seconds=distribution([q['seconds'] for q in requests]),
                    aggregate_output_tokens_per_second=distribution([output_tokens*concurrency/r['seconds'] for r in rounds]),
                    stable_text_per_prompt=all(len(values) == 1 for values in text_sets[engine]))
            a, b = (cell['engines'][name] for name in ('dllm', comparator))
            cell['dllm_over_comparator_throughput_p50'] = a['aggregate_output_tokens_per_second']['p50']/b['aggregate_output_tokens_per_second']['p50']
            cell['dllm_over_comparator_first_output_p95'] = a['first_output_seconds']['p95']/b['first_output_seconds']['p95']
            cell['exact_cross_engine_text'] = all(len(a_text | b_text) == 1
                for a_text, b_text in zip(text_sets['dllm'], text_sets[comparator]))
            rows.append(cell)
    return dict(workload_passed=True, competitive_pass=None, output_tokens=output_tokens, model_revision=first['manifest']['model_revision'],
                hardware=first['manifest']['hardware'], comparator=comparator, rows=rows,
                precision_policies={name: reports[phase]['manifest']['precision']
                                    for name, phase in (('dllm', 'A1'), (comparator, 'B1'))},
                limitation=('Within-dllm policy experiment; numerical/lifecycle gates are separate, no cross-engine verdict.' if candidate else
                            'Engineering comparison only; differing intermediate arithmetic, client-visible timings, no complete internal decode/memory gate.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize(args.directory), indent=2)+'\n')


if __name__ == '__main__':
    main()

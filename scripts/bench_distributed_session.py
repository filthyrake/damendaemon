"""Matched whole-job cost of fresh versus resident two-GPU CLI workers.

This includes Python imports, worker startup, activation, generation and clean
shutdown. It is not a steady-state decode or concurrent HTTP benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import torch
from scripts.verify_partition_service import cases_from_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 12:parser.error('repeats must be 1..12')
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    cases = cases_from_trace(trace)
    selected = [next(case for case in cases if len(case['prompt']) == length)
                for length in (31, 255, 31)]
    precision = trace.get('reference_quantization', 'none')
    if trace['context'] != 512 or precision not in ('none', 'q8_0'):
        parser.error('qualified context and precision required')
    report = dict(passed=False, final_report=False, groups=[], model=trace['path'],
                  quantization=precision, context=512, prompt_lengths=[31, 255, 31],
                  output_tokens_per_request=8, requests_per_job=3,
                  visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  hostname=socket.gethostname(), torch=str(torch.__version__),
                  scope='same-host whole-job ABBA, startup through shutdown; not steady-state decode or HTTP')
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                '--module', 'dllm', 'distributed-run', trace['path'], '--context', '512',
                '--quantization', precision, '--max-tokens', '8', '--ignore-eos', '--json']
    identities = set()
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-session-benchmark-') as directory:
            paths = []
            for index, case in enumerate(selected):
                path = Path(directory)/f'prompt-{index}.json'
                path.write_text(json.dumps(case['prompt']))
                paths.append(path)
            session = Path(directory)/'requests.json'
            session.write_text(json.dumps([dict(tokens=case['prompt']) for case in selected]))
            for mode in ('fresh-workers', 'resident-workers', 'resident-workers', 'fresh-workers'):
                runs = []
                for iteration in range(args.repeats + 1):
                    commands = ([launcher + ['--prompt-tokens-file', str(path)] for path in paths]
                                if mode == 'fresh-workers' else [launcher + ['--requests-file', str(session)]])
                    started = time.perf_counter()
                    completed = [subprocess.run(command, capture_output=True, text=True,
                                                check=True, timeout=180) for command in commands]
                    elapsed = time.perf_counter()-started
                    results = [json.loads(line) for process in completed for line in process.stdout.splitlines()]
                    if len(results) != len(selected):raise AssertionError('missing request result')
                    for index, (actual, case) in enumerate(zip(results, selected)):
                        if (actual['tokens'] != case['tokens'] or actual['finish_reason'] != 'length'
                                or actual['usage'] != dict(prompt_tokens=len(case['prompt']), completion_tokens=8,
                                                           total_tokens=len(case['prompt'])+8)):
                            raise AssertionError('independent reference or usage mismatch')
                        identities.add(actual['model_sha256'])
                        if mode == 'resident-workers' and actual['request_index'] != index:
                            raise AssertionError('session ordering mismatch')
                    if len(identities) != 1:raise AssertionError('prepared weights changed between jobs')
                    if iteration:
                        runs.append(dict(job_seconds=elapsed, results=results))
                    print(json.dumps(dict(mode=mode, iteration=iteration, warmup=iteration == 0,
                                          job_seconds=elapsed)), flush=True)
                report['groups'].append(dict(mode=mode, runs=runs))
                output.write_text(json.dumps(report, indent=2)+'\n')
        report['model_sha256'] = next(iter(identities))
        report['summaries'] = [dict(mode=mode, median_job_seconds=statistics.median(
            run['job_seconds'] for group in report['groups'] if group['mode'] == mode for run in group['runs']))
            for mode in ('fresh-workers', 'resident-workers')]
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['final_report'] = True
        output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':main()

"""Front-door two-GPU CLI checks against independently generated reference tokens."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
from tokenizers import Tokenizer
from scripts.verify_partition_service import cases_from_trace
from scripts.distributed_job import run_job


def main():
    p = argparse.ArgumentParser(); p.add_argument('trace'); p.add_argument('--output', required=True)
    args = p.parse_args()
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    cases = cases_from_trace(trace)
    selected = []
    for length in (31, 255, 496):
        selected.append(next(case for case in cases if len(case['prompt']) == length))
    tokenizer = Tokenizer.from_file(str(Path(trace['path'])/'tokenizer.json'))
    report = dict(passed=False, final_report=False, model=trace['path'],
                  quantization=trace.get('reference_quantization', 'none'), checks=[],
                  scope='actual same-host distributed CLI; correctness/lifecycle, not a performance benchmark')
    output = Path(args.output); output.write_text(json.dumps(report, indent=2)+'\n')
    launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                '--run-path', str(Path(__file__).with_name('distributed_entry.py').resolve()),
                'distributed-run', trace['path'], '--context', '512',
                '--quantization', report['quantization'], '--max-tokens', '8', '--json']
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-distributed-cli-') as directory:
            tokens_file = Path(directory)/'tokens.json'
            for case in selected:
                tokens_file.write_text(json.dumps(case['prompt']))
                result = run_job(launcher + ['--prompt-tokens-file', str(tokens_file), '--ignore-eos'])
                actual = json.loads(result.stdout)
                expected = case['tokens']
                assert actual['tokens'] == expected
                assert actual['text'] == tokenizer.decode(expected, skip_special_tokens=True)
                assert actual['usage'] == dict(prompt_tokens=len(case['prompt']), completion_tokens=8,
                                               total_tokens=len(case['prompt'])+8)
                assert actual['finish_reason'] == 'length'
                report['checks'].append(dict(prompt_tokens=len(case['prompt']), passed=True, result=actual))
            result = run_job(launcher + ['--prompt', 'What is 2 + 2? Answer with just the number.'])
            actual = json.loads(result.stdout)
            assert actual['text'].strip() == '4' and actual['finish_reason'] == 'stop'
            report['checks'].append(dict(kind='chat-eos', passed=True, result=actual))
            # Invalid token IDs must fail at input preparation, not reach GPU execution.
            tokens_file.write_text('[151936]')
            result = run_job(launcher + ['--prompt-tokens-file', str(tokens_file)], check=False)
            assert result.returncode != 0 and not result.stdout.strip()
            assert 'raw prompt must contain valid vocabulary token IDs' in result.stderr
            report['checks'].append(dict(kind='invalid-token', passed=True, exit_code=result.returncode))
            # One resident pair handles changing prompt lengths, EOS and slot reuse.
            session_cases = [selected[i] for i in (2, 0, 1, 0, 2)]
            requests = [dict(tokens=case['prompt'], max_tokens=8, ignore_eos=True) for case in session_cases]
            requests.append(dict(prompt='What is 2 + 2? Answer with just the number.', max_tokens=8))
            requests.append(dict(tokens=selected[0]['prompt'], max_tokens=8, ignore_eos=True))
            session_file = Path(directory)/'requests.json'
            session_file.write_text(json.dumps(requests))
            result = run_job(launcher + ['--requests-file', str(session_file)])
            results = [json.loads(line) for line in result.stdout.splitlines()]
            assert len(results) == 7
            for index, actual in enumerate(results):
                assert actual['request_index'] == index
                assert actual['timing']['release_seconds'] >= 0
                assert (actual['timing']['activation_seconds'] > 0) == (index == 0)
                if index == 5:
                    assert actual['text'].strip() == '4' and actual['finish_reason'] == 'stop'
                else:
                    case = session_cases[index] if index < 5 else selected[0]
                    assert actual['tokens'] == case['tokens']
                    assert actual['text'] == tokenizer.decode(case['tokens'], skip_special_tokens=True)
                    assert actual['usage'] == dict(prompt_tokens=len(case['prompt']), completion_tokens=8,
                                                   total_tokens=len(case['prompt'])+8)
                    assert actual['finish_reason'] == 'length'
            report['checks'].append(dict(kind='resident-session-slot-reuse', passed=True, results=results))
            requests.append(dict(tokens=[151936]))
            session_file.write_text(json.dumps(requests))
            result = run_job(launcher + ['--requests-file', str(session_file)], check=False)
            assert result.returncode != 0 and not result.stdout.strip()
            assert 'raw prompt must contain valid vocabulary token IDs' in result.stderr
            report['checks'].append(dict(kind='invalid-late-request-before-execution', passed=True,
                                         exit_code=result.returncode))
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['final_report'] = True
        output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], checks=len(report['checks']))))


if __name__ == '__main__':main()

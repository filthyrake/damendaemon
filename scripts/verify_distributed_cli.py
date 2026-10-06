"""Front-door two-GPU CLI checks against independently generated reference tokens."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from tokenizers import Tokenizer
from scripts.verify_partition_service import cases_from_trace


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
                '--module', 'dllm', 'distributed-run', trace['path'], '--context', '512',
                '--quantization', report['quantization'], '--max-tokens', '8', '--json']
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-distributed-cli-') as directory:
            tokens_file = Path(directory)/'tokens.json'
            for case in selected:
                tokens_file.write_text(json.dumps(case['prompt']))
                result = subprocess.run(launcher + ['--prompt-tokens-file', str(tokens_file), '--ignore-eos'],
                                        capture_output=True, text=True, timeout=180, check=True)
                actual = json.loads(result.stdout)
                expected = case['tokens']
                assert actual['tokens'] == expected
                assert actual['text'] == tokenizer.decode(expected, skip_special_tokens=True)
                assert actual['usage'] == dict(prompt_tokens=len(case['prompt']), completion_tokens=8,
                                               total_tokens=len(case['prompt'])+8)
                assert actual['finish_reason'] == 'length'
                report['checks'].append(dict(prompt_tokens=len(case['prompt']), passed=True, result=actual))
            result = subprocess.run(launcher + ['--prompt', 'What is 2 + 2? Answer with just the number.'],
                                    capture_output=True, text=True, timeout=180, check=True)
            actual = json.loads(result.stdout)
            assert actual['text'].strip() == '4' and actual['finish_reason'] == 'stop'
            report['checks'].append(dict(kind='chat-eos', passed=True, result=actual))
            # Invalid token IDs must fail at input preparation, not reach GPU execution.
            tokens_file.write_text('[151936]')
            result = subprocess.run(launcher + ['--prompt-tokens-file', str(tokens_file)],
                                    capture_output=True, text=True, timeout=180)
            assert result.returncode != 0 and not result.stdout.strip()
            assert 'raw prompt must contain valid vocabulary token IDs' in result.stderr
            report['checks'].append(dict(kind='invalid-token', passed=True, exit_code=result.returncode))
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['final_report'] = True
        output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], checks=len(report['checks']))))


if __name__ == '__main__':main()

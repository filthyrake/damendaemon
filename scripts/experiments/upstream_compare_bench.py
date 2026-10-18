"""Comparable upstream HTTP benchmark for #1: one server, fixed greedy workload.

Drives K sequential greedy requests against an OpenAI-compatible
/v1/completions endpoint (damendaemon or vLLM serving the identical
checkpoint), measures per-request wall latency and completion tokens, and
reports mean latency plus aggregate round throughput from declared usage.
Both stacks are measured with this same runner so the numbers are comparable.
Latency/throughput on this desktop are comparison evidence only — matched
same-device SLO qualification stays under #54/#63.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.experiments.split_gguf import save

PROMPTS = ['What is 17 * 23? Answer with one number.',
           'Name the capital of France in one word.',
           'Count from one to five with commas.',
           'What color is a banana? One word.',
           'Briefly explain why the sky appears blue.',
           'Write the word elephant backwards.',
           'What is the largest planet in the solar system?',
           'Give one synonym for happy.']
ROUNDS = 2
MAX_TOKENS = 64


def generate(port, model, prompt, timeout=300):
    body = json.dumps(dict(model=model, prompt=prompt, max_tokens=MAX_TOKENS,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:160]}') from None
    elapsed = time.monotonic() - started
    usage = payload.get('usage') or {}
    return dict(prompt=prompt[:32], seconds=elapsed,
                completion_tokens=usage.get('completion_tokens'),
                prompt_tokens=usage.get('prompt_tokens'),
                text=payload['choices'][0]['text'][:60])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--label', required=True, help='stack label recorded in the report')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    requests = []
    for round_index in range(ROUNDS):
        for prompt in PROMPTS:
            requests.append(generate(args.port, args.model, prompt))
    latencies = [entry['seconds'] for entry in requests]
    tokens = [entry['completion_tokens'] or 0 for entry in requests]
    round_seconds = sum(latencies)
    report = dict(label=args.label, port=args.port, model=args.model,
                  prompts=len(PROMPTS), rounds=ROUNDS, max_tokens=MAX_TOKENS,
                  requests=requests,
                  mean_latency_seconds=round(statistics.mean(latencies), 4),
                  median_latency_seconds=round(statistics.median(latencies), 4),
                  total_seconds=round(round_seconds, 4),
                  total_completion_tokens=sum(tokens),
                  aggregate_tokens_per_second=round(sum(tokens) / round_seconds, 4) if round_seconds else None,
                  incomplete=[entry['prompt'] for entry in requests if not entry['completion_tokens']])
    save(args.output, report)
    print(json.dumps({key: report[key] for key in
                      ('label', 'mean_latency_seconds', 'median_latency_seconds',
                       'total_completion_tokens', 'aggregate_tokens_per_second', 'incomplete')}))


if __name__ == '__main__':
    main()

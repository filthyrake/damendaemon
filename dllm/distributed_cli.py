"""Explicit experimental distributed command; model files must already be local."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def prompt_tokens(args, model):
    from .prompts import encode
    if args.prompt_tokens_file is not None:
        with Path(args.prompt_tokens_file).expanduser().open('rb') as source:
            data = source.read(32769)
        if len(data) > 32768:raise ValueError('prompt token file exceeds 32 KiB')
        raw = json.loads(data)
        if not isinstance(raw, list):raise ValueError('prompt token file must contain a JSON token list')
        tokens = encode(model, None, raw_prompt=raw)
    else:
        if args.prompt is None:raise ValueError('coordinator requires --prompt or --prompt-tokens-file')
        if len(args.prompt.encode()) > 65536:raise ValueError('prompt exceeds 64 KiB')
        tokens = encode(model, None if args.raw else [{'role':'user','content':args.prompt}],
                        raw_prompt=args.prompt if args.raw else None)
    if not tokens or len(tokens) + args.max_tokens > args.context:
        raise ValueError('prompt plus max-tokens exceeds distributed context')
    return tokens


def main(argv=None):
    parser = argparse.ArgumentParser(prog='dllm distributed-run',
        description='Experimental two-rank Qwen2.5-0.5B greedy generation under torchrun')
    parser.add_argument('path', help='local matching safetensors snapshot on this worker')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--prompt', help='coordinator chat prompt, or raw text with --raw')
    inputs.add_argument('--prompt-tokens-file', help='coordinator-only JSON list of raw token IDs')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--quantization', choices=('none','q8_0'), default=os.environ.get('DLLM_QUANTIZATION','none'))
    parser.add_argument('--ignore-eos', action='store_true', help='generate the full requested token count')
    parser.add_argument('--json', action='store_true', help='one coordinator result including text, token IDs and timing')
    args = parser.parse_args(argv)
    if not 1 <= args.context <= 512 or not 1 <= args.max_tokens <= args.context:
        parser.error('context must be 1..512 and max-tokens must fit the context')
    os.environ['DLLM_DEVICE'] = 'cuda'
    from .distributed import StagePair
    from .prompts import decode_output
    from .stopping import text_delta
    started = time.perf_counter()
    pair = StagePair(args.path, args.context, args.quantization, lambda model: prompt_tokens(args, model))
    try:
        if pair.rank == 1:
            pair.serve_worker()
            return
        ready = time.perf_counter()
        token = pair.step(pair.prompt)
        first = time.perf_counter()
        output = []
        emitted = ''
        finish = 'length'
        for index in range(args.max_tokens):
            output.append(token)
            stop = not args.ignore_eos and token in pair.model.eos
            delta, emitted, _ = text_delta(decode_output(pair.model, output), emitted,
                                           final=stop or index == args.max_tokens - 1)
            if not args.json:print(delta, end='', flush=True)
            if stop:
                finish = 'stop'
                break
            if index + 1 < args.max_tokens:token = pair.step([token])
        finished = time.perf_counter()
        report = dict(text=emitted, tokens=output, finish_reason=finish,
            model_sha256=pair.identity.hex(), quantization=args.quantization,
            scope='experimental two-rank single-request greedy generation',
            usage=dict(prompt_tokens=len(pair.prompt), completion_tokens=len(output),
                       total_tokens=len(pair.prompt)+len(output)),
            timing=dict(activation_seconds=ready-started, first_token_seconds=first-started,
                        generation_first_token_seconds=first-ready, generation_seconds=finished-ready,
                        total_seconds=finished-started,
                        decode_tokens_per_second=(len(output)-1)/(finished-first) if len(output)>1 else None))
        pair.stop()
        if args.json:print(json.dumps(report))
        else:
            print()
            print(json.dumps({key:value for key,value in report.items() if key not in ('text','tokens')}), file=sys.stderr)
    finally:
        try:pair.stop()
        finally:pair.close()

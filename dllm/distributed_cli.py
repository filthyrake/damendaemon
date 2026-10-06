"""Explicit experimental distributed command; model files must already be local."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def prepare_requests(args, model):
    """Validate the entire bounded session before either worker starts serving."""
    from .prompts import encode
    if args.requests_file is None:
        return [dict(tokens=prompt_tokens(args, model), max_tokens=args.max_tokens,
                     ignore_eos=args.ignore_eos)]
    with Path(args.requests_file).expanduser().open('rb') as source:
        data = source.read(1048577)
    if len(data) > 1048576:raise ValueError('requests file exceeds 1 MiB')
    rows = json.loads(data)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError('requests file must contain 1..64 request objects')
    prepared = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) - {'prompt', 'tokens', 'raw', 'max_tokens', 'ignore_eos'}
                or ('prompt' in row) == ('tokens' in row)):
            raise ValueError('each request requires exactly one prompt or tokens field and known options')
        raw = row.get('raw', args.raw)
        ignore_eos = row.get('ignore_eos', args.ignore_eos)
        maximum = row.get('max_tokens', args.max_tokens)
        if type(raw) is not bool or type(ignore_eos) is not bool:
            raise ValueError('raw and ignore_eos must be booleans')
        if type(maximum) is not int or not 1 <= maximum <= args.context:
            raise ValueError('request max_tokens must fit the context')
        if 'tokens' in row:
            if not isinstance(row['tokens'], list):raise ValueError('tokens must be a JSON token list')
            if 'raw' in row:raise ValueError('raw applies only to text prompts')
            tokens = encode(model, None, raw_prompt=row['tokens'])
        else:
            prompt = row['prompt']
            if not isinstance(prompt, str) or len(prompt.encode()) > 65536:
                raise ValueError('request prompt must be text of at most 64 KiB')
            tokens = encode(model, None if raw else [{'role':'user', 'content':prompt}],
                            raw_prompt=prompt if raw else None)
        if not tokens or len(tokens) + maximum > args.context:
            raise ValueError('prompt plus max-tokens exceeds distributed context')
        prepared.append(dict(tokens=tokens, max_tokens=maximum, ignore_eos=ignore_eos))
    return prepared


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
    inputs.add_argument('--requests-file', help='coordinator-only JSON array of 1..64 sequential requests; emits JSONL')
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
    pair = StagePair(args.path, args.context, args.quantization, lambda model: prepare_requests(args, model))
    try:
        if pair.rank == 1:
            pair.serve_worker()
            return
        ready = time.perf_counter()
        json_output = args.json or args.requests_file is not None
        for request_index, request in enumerate(pair.requests):
            request_started = ready if request_index == 0 else time.perf_counter()
            token = pair.step(request['tokens'])
            first = time.perf_counter()
            output = []
            emitted = ''
            finish = 'length'
            for index in range(request['max_tokens']):
                output.append(token)
                stop = not request['ignore_eos'] and token in pair.model.eos
                delta, emitted, _ = text_delta(decode_output(pair.model, output), emitted,
                                               final=stop or index == request['max_tokens'] - 1)
                if not json_output:print(delta, end='', flush=True)
                if stop:
                    finish = 'stop'
                    break
                if index + 1 < request['max_tokens']:token = pair.step([token])
            finished = time.perf_counter()
            report = dict(text=emitted, tokens=output, finish_reason=finish,
                model_sha256=pair.identity.hex(), quantization=args.quantization,
                scope='experimental two-rank sequential greedy generation',
                usage=dict(prompt_tokens=len(request['tokens']), completion_tokens=len(output),
                           total_tokens=len(request['tokens'])+len(output)),
                timing=dict(activation_seconds=ready-started if request_index == 0 else 0,
                            first_token_seconds=first-(started if request_index == 0 else request_started),
                            generation_first_token_seconds=first-request_started,
                            generation_seconds=finished-request_started,
                            total_seconds=finished-(started if request_index == 0 else request_started),
                            decode_tokens_per_second=(len(output)-1)/(finished-first) if len(output)>1 else None))
            if args.requests_file is not None:
                release_started = time.perf_counter()
                pair.release()
                report['request_index'] = request_index
                report['timing']['release_seconds'] = time.perf_counter()-release_started
            else:
                pair.stop()
            if json_output:print(json.dumps(report), flush=True)
            else:
                print()
                print(json.dumps({key:value for key,value in report.items() if key not in ('text','tokens')}), file=sys.stderr)
        pair.stop()
    finally:
        try:pair.stop()
        finally:pair.close()

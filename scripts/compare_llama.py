"""Compare warm HTTP latency with an existing isolated llama.cpp endpoint.

Both servers must use the same weights/precision and context. This client
disables llama.cpp prompt caching explicitly. Run servers on the GPU sequentially.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import sys
import time
from urllib.request import Request, urlopen

from dllm.bench import events, _response_json, _tool_call_output
from dllm.token_workload import load_token_prompt


def _score_count(logprobs, requested_top):
    """Count validated score records, never arbitrary container/string length."""
    if logprobs is None:
        return 0
    if not isinstance(logprobs, dict):
        raise ValueError('invalid comparison logprob envelope')
    content = logprobs.get('content')
    if content is None:
        return 0
    if not isinstance(content, list):
        raise ValueError('invalid comparison logprob content')

    def record(value):
        if (not isinstance(value, dict) or not isinstance(value.get('token'), str)
                or type(value.get('logprob')) not in (int, float)
                or not -sys.float_info.max <= value['logprob'] <= 0):
            raise ValueError('invalid comparison token probability')
        raw = value.get('bytes')
        if raw is not None and (not isinstance(raw, list)
                or any(type(byte) is not int or not 0<=byte<=255 for byte in raw)):
            raise ValueError('invalid comparison token bytes')

    for value in content:
        record(value)
        top = value.get('top_logprobs', [])
        maximum = 20 if requested_top is None else requested_top
        if (not isinstance(top, list) or len(top)>maximum
                or (requested_top and not top)):
            raise ValueError('invalid or missing comparison top probabilities')
        for candidate in top:
            record(candidate)
    return len(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--engine", choices=["dllm", "llama.cpp", "vllm"], required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    parser.add_argument('--raw', action='store_true', help='benchmark a raw code-completion prompt')
    parser.add_argument('--prompt-token-ids', help='JSON token array, identical across engines; uses raw completion API')
    parser.add_argument('--top-logprobs',type=int,choices=range(21),default=None)
    parser.add_argument('--max-tokens',type=int,default=64)
    args = parser.parse_args()
    if args.max_tokens<2:parser.error('max tokens must be at least two')
    prompt_ids = None
    if args.prompt_token_ids:
        try:prompt_ids = load_token_prompt(args.prompt_token_ids)
        except (OSError,ValueError):parser.error('invalid or oversized prompt token file')
        args.raw = True
    if not 1 <= args.concurrency <= 8:
        parser.error("concurrency must be 1..8")
    if args.raw and args.top_logprobs is not None:
        parser.error('logprob benchmark currently requires chat completion')
    def request(trial):
        body = {"model": "a", "messages": [{"role": "user", "content":
                "Explain how matrix multiplication works in detail, with a worked example."}],
                "temperature": 0, "max_tokens": args.max_tokens, "stream": True,
                "stream_options": {"include_usage": True}}
        if args.raw:
            del body['messages']
            body['prompt'] = prompt_ids if prompt_ids is not None else 'def binary_search(values, target):\n    '
        if args.thinking != "auto":
            body["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
        if args.engine == "llama.cpp":
            body["cache_prompt"] = False
        if args.top_logprobs is not None:
            body.update(logprobs=True,top_logprobs=args.top_logprobs)
        start = time.perf_counter()
        first = None
        usage = None
        text = ""
        done = False
        finished = False
        scored_tokens=0
        with urlopen(Request(args.base_url + ("/completions" if args.raw else "/chat/completions"), data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as response:
            for data in events(response, require_terminated=True):
                if data == "[DONE]":
                    if not finished:
                        raise ValueError('comparison stream ended without a completion finish')
                    done = True
                    break
                chunk = _response_json(data)
                if not isinstance(chunk, dict):
                    raise ValueError('comparison stream event must be an object')
                if chunk.get("error") is not None:
                    raise ValueError("server returned an error event")
                choices = chunk.get('choices', [])
                if (not isinstance(choices, list) or len(choices)>1
                        or any(not isinstance(choice, dict) for choice in choices)):
                    raise ValueError('comparison requires a single completion choice')
                for choice in choices:
                    if not args.raw and not isinstance(choice.get('delta', {}), dict):
                        raise ValueError('invalid comparison completion delta')
                    content = choice.get('text') if args.raw else choice.get('delta', {}).get('content')
                    if content is None:
                        content = ''
                    if not isinstance(content, str):
                        raise ValueError('invalid comparison completion text')
                    semantic_output = bool(content)
                    if not args.raw:
                        delta = choice.get('delta', {})
                        if any(delta.get(key) is not None and not isinstance(delta[key], str)
                               for key in ('reasoning', 'reasoning_content')):
                            raise ValueError('invalid comparison reasoning delta')
                        tool_output = _tool_call_output(delta.get('tool_calls'))
                        semantic_output = semantic_output or tool_output or any(
                            delta.get(key) for key in ('reasoning', 'reasoning_content'))
                    scores = _score_count(choice.get('logprobs'), args.top_logprobs)
                    if finished and (semantic_output or scores):
                        raise ValueError('comparison output received after completion finish')
                    reason = choice.get('finish_reason')
                    if reason is not None:
                        if not isinstance(reason, str) or not reason or finished:
                            raise ValueError('invalid or duplicate comparison completion finish')
                        finished = True
                    scored_tokens += scores
                    if scored_tokens > args.max_tokens:
                        raise ValueError('comparison scores exceed completion token budget')
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        text += content
                reported_usage = chunk.get("usage")
                if reported_usage is not None:
                    if not isinstance(reported_usage, dict):
                        raise ValueError('invalid comparison usage metadata')
                    tokens = reported_usage.get('completion_tokens')
                    if tokens is not None:
                        if type(tokens) is not int or not 0 <= tokens <= args.max_tokens:
                            raise ValueError('invalid comparison completion token usage')
                        if usage is not None and tokens != usage['completion_tokens']:
                            raise ValueError('conflicting comparison completion token usage')
                        usage = reported_usage
        elapsed = time.perf_counter() - start
        if (not done or first is None or not isinstance(usage, dict)
                or type(usage.get("completion_tokens")) is not int
                or usage["completion_tokens"] != args.max_tokens):
            raise ValueError('comparison requires complete output and exact integer token usage')
        if args.top_logprobs is not None and scored_tokens <= 0:
            raise ValueError('comparison requires requested logprob evidence')
        return {"warmup": trial == 0, "total_seconds": elapsed,
                       "first_output_seconds": first - start,
                       "post_first_output_tokens_per_second": (args.max_tokens-1) / (elapsed - (first - start)),
                       "usage": usage, "text": text,'scored_tokens':scored_tokens}
    trials, rounds = [], []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for trial in range(6):
            start = time.perf_counter()
            results = list(pool.map(request, [trial] * args.concurrency))
            elapsed = time.perf_counter() - start
            trials.extend(results)
            rounds.append({"warmup": trial == 0, "seconds": elapsed,
                           "aggregate_output_tokens_per_second": args.max_tokens * args.concurrency / elapsed})
    print(json.dumps({"engine": args.engine, "concurrency": args.concurrency, "thinking": args.thinking, 'raw': args.raw,
                      'prompt_token_ids': prompt_ids,'top_logprobs':args.top_logprobs,'max_tokens':args.max_tokens, "trials": trials, "rounds": rounds}, indent=2))


if __name__ == "__main__":
    main()

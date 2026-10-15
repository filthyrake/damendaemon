"""Compare warm HTTP latency with an existing isolated llama.cpp endpoint.

Both servers must use the same weights/precision and context. This client
disables llama.cpp prompt caching explicitly. Run servers on the GPU sequentially.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import time
from urllib.request import Request, urlopen

from dllm.bench import events


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
        with open(args.prompt_token_ids) as source:
            prompt_ids = json.load(source)
        if not isinstance(prompt_ids,list) or not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            parser.error('prompt token file must contain a nonempty integer array')
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
        scored_tokens=0
        with urlopen(Request(args.base_url + ("/completions" if args.raw else "/chat/completions"), data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as response:
            for data in events(response, require_terminated=True):
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                for choice in chunk.get("choices", []):
                    scored_tokens+=len((choice.get('logprobs') or {}).get('content') or [])
                    content = choice.get('text', '') if args.raw else choice.get("delta", {}).get("content", "")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        text += content
                if chunk.get("usage"):
                    usage = chunk["usage"]
        elapsed = time.perf_counter() - start
        assert done and first and usage and usage["completion_tokens"] == args.max_tokens
        if args.top_logprobs is not None:assert scored_tokens>0
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

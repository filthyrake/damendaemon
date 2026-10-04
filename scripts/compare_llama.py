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
    args = parser.parse_args()
    prompt_ids = None
    if args.prompt_token_ids:
        with open(args.prompt_token_ids) as source:
            prompt_ids = json.load(source)
        if not isinstance(prompt_ids,list) or not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            parser.error('prompt token file must contain a nonempty integer array')
        args.raw = True
    if not 1 <= args.concurrency <= 8:
        parser.error("concurrency must be 1..8")
    def request(trial):
        body = {"model": "a", "messages": [{"role": "user", "content":
                "Explain how matrix multiplication works in detail, with a worked example."}],
                "temperature": 0, "max_tokens": 64, "stream": True,
                "stream_options": {"include_usage": True}}
        if args.raw:
            del body['messages']
            body['prompt'] = prompt_ids if prompt_ids is not None else 'def binary_search(values, target):\n    '
        if args.thinking != "auto":
            body["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
        if args.engine == "llama.cpp":
            body["cache_prompt"] = False
        start = time.perf_counter()
        first = None
        usage = None
        text = ""
        done = False
        with urlopen(Request(args.base_url + ("/completions" if args.raw else "/chat/completions"), data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as response:
            for data in events(response):
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                for choice in chunk.get("choices", []):
                    content = choice.get('text', '') if args.raw else choice.get("delta", {}).get("content", "")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        text += content
                if chunk.get("usage"):
                    usage = chunk["usage"]
        elapsed = time.perf_counter() - start
        assert done and first and usage and usage["completion_tokens"] == 64
        return {"warmup": trial == 0, "total_seconds": elapsed,
                       "first_output_seconds": first - start,
                       "post_first_output_tokens_per_second": 63 / (elapsed - (first - start)),
                       "usage": usage, "text": text}
    trials, rounds = [], []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for trial in range(6):
            start = time.perf_counter()
            results = list(pool.map(request, [trial] * args.concurrency))
            elapsed = time.perf_counter() - start
            trials.extend(results)
            rounds.append({"warmup": trial == 0, "seconds": elapsed,
                           "aggregate_output_tokens_per_second": 64 * args.concurrency / elapsed})
    print(json.dumps({"engine": args.engine, "concurrency": args.concurrency, "thinking": args.thinking, 'raw': args.raw,
                      'prompt_token_ids': prompt_ids, "trials": trials, "rounds": rounds}, indent=2))


if __name__ == "__main__":
    main()

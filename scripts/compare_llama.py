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
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 8:
        parser.error("concurrency must be 1..8")
    def request(trial):
        body = {"model": "a", "messages": [{"role": "user", "content":
                "Explain how matrix multiplication works in detail, with a worked example."}],
                "temperature": 0, "max_tokens": 64, "stream": True,
                "stream_options": {"include_usage": True}}
        if args.engine == "llama.cpp":
            body["cache_prompt"] = False
        start = time.perf_counter()
        first = None
        usage = None
        text = ""
        done = False
        with urlopen(Request(args.base_url + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"}), timeout=120) as response:
            for data in events(response):
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                for choice in chunk.get("choices", []):
                    content = choice.get("delta", {}).get("content", "")
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
    print(json.dumps({"engine": args.engine, "concurrency": args.concurrency,
                      "trials": trials, "rounds": rounds}, indent=2))


if __name__ == "__main__":
    main()

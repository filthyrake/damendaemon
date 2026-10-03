"""Dependency-free, closed-loop streaming benchmark for compatible HTTP APIs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


@dataclass
class Result:
    index: int
    model: str
    seconds: float
    first_output_seconds: float | None = None
    output_tokens: int | None = None
    error: str | None = None


def events(lines):
    """Decode SSE data fields; comments and non-data fields aren't payloads."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def request_one(base_url, model, prompt, max_tokens, timeout, index, api_key=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(base_url.rstrip("/") + "/chat/completions",
                  data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    result = Result(index=index, model=model, seconds=0)
    done = False
    try:
        with urlopen(req, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("expected text/event-stream response")
            for data in events(response):
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if not isinstance(chunk, dict):
                    raise ValueError("stream event must be an object")
                if chunk.get("error"):
                    raise ValueError("server returned an error event")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if any(delta.get(key) for key in
                           ("content", "reasoning_content", "reasoning", "tool_calls")):
                        if result.first_output_seconds is None:
                            result.first_output_seconds = time.perf_counter() - start
                usage = chunk.get("usage") or {}
                tokens = usage.get("completion_tokens")
                if tokens is not None:
                    if type(tokens) is not int or tokens < 0:
                        raise ValueError("invalid completion_tokens usage")
                    result.output_tokens = tokens
        if not done:
            raise ValueError("stream ended without [DONE]")
    except HTTPError as exc:
        # Do not echo response bodies, prompts, credentials, or URLs.
        result.error = f"HTTP {exc.code}"
        exc.close()
    except Exception as exc:
        result.error = type(exc).__name__
    result.seconds = time.perf_counter() - start
    return result


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"p50": statistics.median(ordered),
            "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]}


def summarize(results, elapsed):
    good = [result for result in results if result.error is None]
    complete_usage = bool(good) and all(r.output_tokens is not None for r in good)
    tokens = sum(r.output_tokens for r in good) if complete_usage else None
    return {
        "requests": len(results),
        "succeeded": len(good),
        "failed": len(results) - len(good),
        "wall_seconds": elapsed,
        "successful_requests_per_second": len(good) / elapsed if elapsed > 0 else None,
        "successful_output_tokens": tokens,
        "successful_output_tokens_per_second": tokens / elapsed
        if tokens is not None and elapsed > 0 else None,
        "latency_seconds": distribution([r.seconds for r in good]),
        "first_output_seconds": distribution([
            r.first_output_seconds for r in good if r.first_output_seconds is not None
        ]),
        "requests_with_first_output": sum(r.first_output_seconds is not None for r in good),
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="measure a streaming chat endpoint")
    bench.add_argument("--base-url", default="http://localhost:8000/v1")
    bench.add_argument("--models", nargs="+", required=True,
                       help="cycle models in submission order; use concurrency 1 for switches")
    bench.add_argument("--requests", type=positive_int, default=8)
    bench.add_argument("--concurrency", type=positive_int, default=1)
    bench.add_argument("--max-tokens", type=positive_int, default=128)
    bench.add_argument("--timeout", type=positive_float, default=120,
                       help="socket inactivity timeout in seconds, not a total request deadline")
    bench.add_argument("--prompt-file", type=Path)
    bench.add_argument("--label", default="unlabeled")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        parser.error("base URL must be HTTP(S), without credentials, query, or fragment")
    try:
        prompt = args.prompt_file.read_text() if args.prompt_file else (
            "Explain how a GPU executes matrix multiplication in clear, detailed terms."
        )
    except OSError as exc:
        parser.error(f"cannot read prompt file: {exc.strerror}")
    api_key = os.environ.get("DLLM_API_KEY")
    start = time.perf_counter()
    results = []
    # Keep only concurrency requests submitted at once, avoiding an unbounded
    # future queue and making request-level timing match admitted client load.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = set()
        next_index = 0
        while next_index < args.requests or pending:
            while next_index < args.requests and len(pending) < args.concurrency:
                pending.add(pool.submit(
                    request_one, args.base_url, args.models[next_index % len(args.models)],
                    prompt, args.max_tokens, args.timeout, next_index, api_key,
                ))
                next_index += 1
            future = next(as_completed(pending))
            pending.remove(future)
            results.append(future.result())
    elapsed = time.perf_counter() - start
    report = {
        "schema_version": 1,
        "label": args.label,
        "workload": {
            "base_url": args.base_url, "models": args.models,
            "concurrency": args.concurrency, "max_tokens": args.max_tokens,
            "temperature": 0, "socket_timeout_seconds": args.timeout,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "warmup_requests": 0,
        },
        "summary": summarize(results, elapsed),
        "results": [asdict(r) for r in sorted(results, key=lambda r: r.index)],
    }
    print(json.dumps(report, indent=2))
    if any(r.error is not None for r in results):
        raise SystemExit(1)

"""Bounded closed-loop or fixed-arrival streaming HTTP benchmark."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
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


MAX_OUTPUT_EVENTS = 100_000

def llama_context(base_url, expected):
    """Fail before timing if the comparator silently capped its KV slot context."""
    with urlopen(base_url.rstrip('/') + '/props', timeout=10) as response:
        props = json.load(response)
    actual = props.get('default_generation_settings', {}).get('n_ctx')
    if type(actual) is not int or actual != expected:
        raise RuntimeError(f'llama.cpp slot context mismatch: requested {expected}, actual {actual}')
    return {'requested_context': expected, 'actual_context': actual,
            'total_slots': props.get('total_slots'), 'model_ftype': props.get('model_ftype')}


@dataclass
class Result:
    index: int
    model: str
    seconds: float
    first_output_seconds: float | None = None
    output_tokens: int | None = None
    error: str | None = None
    output_events: int = 0
    last_output_seconds: float | None = None
    output_event_gaps_seconds: list = field(default_factory=list)
    scheduled_offset_seconds: float | None = None
    dispatch_delay_seconds: float | None = None
    reported_server_queue_seconds: float | None = None


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
                has_output = any(
                    any(choice.get("delta", {}).get(key) for key in
                        ("content", "reasoning_content", "reasoning", "tool_calls"))
                    for choice in chunk.get("choices", []))
                if has_output:
                    if result.output_events >= MAX_OUTPUT_EVENTS:
                        raise ValueError("output event sample limit exceeded")
                    arrived = time.perf_counter() - start
                    if result.first_output_seconds is None:
                        result.first_output_seconds = arrived
                    else:
                        result.output_event_gaps_seconds.append(arrived-result.last_output_seconds)
                    result.last_output_seconds = arrived
                    result.output_events += 1
                timing = chunk.get("dllm_timing") or {}
                if not isinstance(timing, dict):
                    raise ValueError("invalid server timing metadata")
                queued = timing.get("queue_seconds")
                if queued is not None:
                    if type(queued) not in (int, float) or not 0 <= queued < float('inf'):
                        raise ValueError("invalid reported queue time")
                    result.reported_server_queue_seconds = queued
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
            "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            "p99": ordered[max(0, math.ceil(len(ordered) * 0.99) - 1)]}


def summarize(results, elapsed):
    good = [result for result in results if result.error is None]
    complete_usage = bool(good) and all(r.output_tokens is not None for r in good)
    tokens = sum(r.output_tokens for r in good) if complete_usage else None
    return {
        "requests": len(results),
        "succeeded": len(good),
        "failed": len(results) - len(good),
        "client_capacity_drops": sum(r.error == "client capacity" for r in results),
        "http_failures": sum(bool(r.error and r.error.startswith("HTTP ")) for r in results),
        "wall_seconds": elapsed,
        "successful_requests_per_second": len(good) / elapsed if elapsed > 0 else None,
        "successful_output_tokens": tokens,
        "successful_output_tokens_per_second": tokens / elapsed
        if tokens is not None and elapsed > 0 else None,
        "latency_seconds": distribution([r.seconds for r in good]),
        "first_output_seconds": distribution([
            r.first_output_seconds for r in good if r.first_output_seconds is not None
        ]),
        "output_event_gaps_seconds": distribution([
            gap for r in good for gap in r.output_event_gaps_seconds]),
        "max_output_event_gap_seconds_per_request": distribution([
            max(r.output_event_gaps_seconds) for r in good if r.output_event_gaps_seconds]),
        "requests_with_output_gaps": sum(bool(r.output_event_gaps_seconds) for r in good),
        "dispatch_delay_seconds": distribution([
            r.dispatch_delay_seconds for r in results if r.dispatch_delay_seconds is not None]),
        "reported_server_queue_seconds": distribution([
            r.reported_server_queue_seconds for r in good if r.reported_server_queue_seconds is not None]),
        "requests_with_server_queue_timing": sum(r.reported_server_queue_seconds is not None for r in good),
        "requests_with_first_output": sum(r.first_output_seconds is not None for r in good),
    }


def evaluate_profile(result, profile):
    """Judge measured client behavior; never infer tokens from SSE events."""
    if result.error is not None:
        return {"status": "error", "reasons": [result.error]}
    missing = []
    if result.first_output_seconds is None:
        missing.append("first output unavailable")
    if not result.output_event_gaps_seconds:
        missing.append("output event gaps unavailable")
    if result.output_tokens is None or result.seconds <= 0:
        missing.append("request-average token rate unavailable")
    if missing:
        return {"status": "unmeasurable", "reasons": missing}
    rate = result.output_tokens/result.seconds
    maximum_gap = max(result.output_event_gaps_seconds)
    reasons = []
    if result.first_output_seconds > profile['max_first_output_seconds']:
        reasons.append('first output budget exceeded')
    if maximum_gap > profile['max_output_event_gap_seconds']:
        reasons.append('output event gap budget exceeded')
    if rate < profile['min_request_output_tokens_per_second']:
        reasons.append('request-average token rate below budget')
    return {"status": "failed" if reasons else "passed", "reasons": reasons,
            "request_output_tokens_per_second": rate,
            "maximum_output_event_gap_seconds": maximum_gap}


def profile_summary(results, elapsed, profile):
    evaluations = [(result, evaluate_profile(result, profile)) for result in results]
    passed = [result for result, evaluation in evaluations if evaluation['status'] == 'passed']
    tokens = sum(result.output_tokens for result in passed)
    return {
        "measurement_scope": "client SSE gaps and whole-request average token rate; not GPU per-token SLO",
        "budgets": profile,
        "all_requests_passed": bool(results) and len(passed) == len(results),
        "counts": {state: sum(evaluation['status'] == state for _, evaluation in evaluations)
                   for state in ('passed', 'failed', 'unmeasurable', 'error')},
        "qualifying_requests_per_second": len(passed)/elapsed if elapsed > 0 else None,
        "qualifying_output_tokens": tokens,
        "qualifying_output_tokens_per_second": tokens/elapsed if elapsed > 0 else None,
        "results": [{"index": result.index, **evaluation} for result, evaluation in
                    sorted(evaluations, key=lambda pair: pair[0].index)],
    }


def fixed_arrivals(send, models, requests, concurrency, rate):
    """Offer each scheduled arrival once; never queue behind busy client slots."""
    started = time.perf_counter()
    results = []
    def dispatch(index, model, scheduled):
        delay = max(0, time.perf_counter()-started-scheduled)
        result = send(index, model)
        result.scheduled_offset_seconds = scheduled
        result.dispatch_delay_seconds = delay
        return result
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = set()
        for index in range(requests):
            scheduled = index/rate
            delay = started+scheduled-time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            completed = {future for future in pending if future.done()}
            for future in completed:
                results.append(future.result())
            pending -= completed
            model = models[index % len(models)]
            if len(pending) >= concurrency:
                results.append(Result(index, model, 0, error="client capacity",
                    scheduled_offset_seconds=scheduled))
            else:
                pending.add(pool.submit(dispatch, index, model, scheduled))
        for future in as_completed(pending):
            results.append(future.result())
    return results


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
    bench.add_argument("--arrival-rate", type=positive_float,
                       help="fixed offered requests/second; concurrency caps client in-flight requests")
    bench.add_argument("--max-tokens", type=positive_int, default=128)
    bench.add_argument("--timeout", type=positive_float, default=120,
                       help="socket inactivity timeout in seconds, not a total request deadline")
    bench.add_argument("--slo-first-output", type=positive_float,
                       help="maximum client first-output seconds; requires both other --slo options")
    bench.add_argument("--slo-output-gap", type=positive_float,
                       help="maximum client SSE output-event gap seconds, not per-token latency")
    bench.add_argument("--slo-request-rate", type=positive_float,
                       help="minimum reported output tokens / whole request seconds, including prefill")
    bench.add_argument("--prompt-file", type=Path)
    bench.add_argument("--label", default="unlabeled")
    args = parser.parse_args()
    budgets = (args.slo_first_output, args.slo_output_gap, args.slo_request_rate)
    if any(value is not None for value in budgets) and not all(value is not None for value in budgets):
        parser.error('a client profile requires all three --slo options')
    profile = (dict(max_first_output_seconds=args.slo_first_output,
                    max_output_event_gap_seconds=args.slo_output_gap,
                    min_request_output_tokens_per_second=args.slo_request_rate)
               if all(value is not None for value in budgets) else None)
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
    if args.arrival_rate is not None:
        results = fixed_arrivals(
            lambda index, model: request_one(args.base_url, model, prompt, args.max_tokens,
                                            args.timeout, index, api_key),
            args.models, args.requests, args.concurrency, args.arrival_rate)
    else:
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
        "schema_version": 2,
        "label": args.label,
        "workload": {
            "base_url": args.base_url, "models": args.models,
            "arrival_mode": "fixed_rate" if args.arrival_rate is not None else "closed_loop",
            "offered_requests_per_second": args.arrival_rate,
            "output_timing_unit": "SSE output event, not token",
            "max_output_events_per_request": MAX_OUTPUT_EVENTS,
            "concurrency": args.concurrency, "max_tokens": args.max_tokens,
            "temperature": 0, "socket_timeout_seconds": args.timeout,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "warmup_requests": 0,
        },
        "summary": summarize(results, elapsed),
        "results": [asdict(r) for r in sorted(results, key=lambda r: r.index)],
    }
    if profile is not None:
        report['client_profile'] = profile_summary(results, elapsed, profile)
    print(json.dumps(report, indent=2))
    if (any(r.error is not None for r in results)
            or (profile is not None and not report['client_profile']['all_requests_passed'])):
        raise SystemExit(1)

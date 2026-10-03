"""Exercise the real CLI/server, model transitions, streaming, and recovery."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dllm.bench import events, request_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    args = parser.parse_args()
    prompt = "Explain how matrix multiplication works in detail, with a worked example."
    reports = []
    for slots in (1, 2):
        with tempfile.TemporaryDirectory(prefix="dllm-proof-") as directory:
            log_path = Path(directory) / "server.log"
            with log_path.open("w") as log:
                process = subprocess.Popen([
                    sys.executable, "-m", "dllm", "serve", "--port", "0", "--context", "256",
                    "--gpu-models", str(slots), "--model", "a=" + args.model_a,
                    "--model", "b=" + args.model_b, "--model", "broken=/missing-dllm-model",
                ], stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 90
                    base = None
                    while time.monotonic() < deadline and process.poll() is None:
                        lines = log_path.read_text().splitlines()
                        ready = [line for line in lines if line.startswith("dllm listening on ")]
                        if ready:
                            base = ready[-1].split()[-1]
                            break
                        time.sleep(0.1)
                    if base is None:
                        raise RuntimeError("server did not start: " + log_path.read_text())
                    def post(alias="a", **extra):
                        data = {"model": alias, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 64, "temperature": 0}
                        data.update(extra)
                        return urlopen(Request(base + "/v1/chat/completions", data=json.dumps(data).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=120)
                    trials = []
                    texts = {}
                    for alias in ("a", "b", "a", "b", "a"):
                        start = time.perf_counter()
                        with post(alias) as response:
                            data = json.load(response)
                        text = data["choices"][0]["message"]["content"]
                        if alias in texts:
                            assert text == texts[alias], "model switch changed deterministic output"
                        texts[alias] = text
                        trials.append({"model": alias, "wall_seconds": time.perf_counter() - start,
                                       "usage": data["usage"], "timing": data["dllm_timing"]})
                    expected_tier = "host" if slots == 1 else "gpu"
                    assert trials[2]["timing"]["source_tier"] == expected_tier
                    # Streaming matches the non-streaming result and reports real usage.
                    with post(stream=True, stream_options={"include_usage": True}) as response:
                        chunks = list(events(response))
                    assert chunks[-1] == "[DONE]"
                    parsed = [json.loads(value) for value in chunks[:-1]]
                    streamed = "".join(c["delta"].get("content", "") for item in parsed for c in item["choices"])
                    assert streamed == texts["a"]
                    assert parsed[-1]["usage"]["completion_tokens"] == trials[0]["usage"]["completion_tokens"]
                    # Invalid requests and load failures cannot poison the worker.
                    for alias, extra, status in (("a", {"max_tokens": 10000}, 400), ("broken", {}, 500)):
                        try:
                            post(alias, **extra).close()
                            raise AssertionError("bad request unexpectedly succeeded")
                        except HTTPError as error:
                            assert error.code == status, error.code
                            error.close()
                    # Close mid-generation and immediately ask for another model.
                    with post(stream=True) as response:
                        next(events(response))
                    with post("b") as response:
                        assert json.load(response)["choices"][0]["message"]["content"] == texts["b"]
                    def concurrent(alias):
                        with post(alias) as response:
                            return alias, json.load(response)["choices"][0]["message"]["content"]
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for alias, text in pool.map(concurrent, ("a", "b")):
                            assert text == texts[alias]
                    benchmark = request_one(base + "/v1", "a", prompt, 64, 120, 0)
                    assert benchmark.error is None, benchmark.error
                    reports.append({"gpu_models": slots, "trials": trials, "sample_text": texts,
                                    "streaming_and_recovery_passed": True,
                                    "stream_benchmark": vars(benchmark)})
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    print(json.dumps({"model_a": args.model_a, "model_b": args.model_b, "reports": reports}, indent=2))


if __name__ == "__main__":
    main()

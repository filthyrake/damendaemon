"""Exercise the real CLI/server, model transitions, streaming, and recovery."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
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
    parser.add_argument('--context',type=int,default=256)
    parser.add_argument('--swap-cycles',type=int,default=2)
    parser.add_argument('--disable-thinking',action='store_true')
    parser.add_argument('--prefill-chunk-size', type=int, default=0)
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--host-gib', type=float, default=8)
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--output',help='persist report and per-capacity server logs')
    parser.add_argument('--candidate-q8-small-batch', action='store_true',
                        help='start each server with the isolated HOMER Q8 experiment')
    args = parser.parse_args()
    if not 128 <= args.context <= 4096 or not 1 <= args.swap_cycles <= 20:
        parser.error('context must be 128..4096 and swap cycles 1..20')
    if args.prefill_chunk_size < 0:
        parser.error('prefill chunk size must be nonnegative')
    import math
    if any(not math.isfinite(value) or value <= 0 for value in (args.host_gib, args.request_timeout)):
        parser.error('host GiB and request timeout must be finite and positive')
    report=dict(passed=False,model_a=args.model_a,model_b=args.model_b,context=args.context,
                swap_cycles=args.swap_cycles,reports=[],
                prefill_chunk_size=args.prefill_chunk_size,
                graph_requested=not args.no_graph, host_gib=args.host_gib,
                request_timeout=args.request_timeout,
                hybrid_precision=os.environ.get('DLLM_HYBRID_PRECISION', 'float16'),
                rocm_aotriton_experimental=os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'),
                variant='hip-q8-small-batch' if args.candidate_q8_small_batch else 'ordinary')
    prompt = "Explain how matrix multiplication works in detail, with a worked example."
    reports = report["reports"]
    for slots in (1, 2):
        with tempfile.TemporaryDirectory(prefix="dllm-proof-") as directory:
            log_path = (Path(args.output).with_suffix(f'.capacity{slots}.log')
                        if args.output else Path(directory) / "server.log")
            with log_path.open("w") as log:
                entry = ([sys.executable, 'scripts/experiments/hip_q8_small_batch.py', '--module', 'dllm']
                         if args.candidate_q8_small_batch else [sys.executable, '-m', 'dllm'])
                process = subprocess.Popen([
                    *entry, "serve", "--port", "0", "--context", str(args.context),
                    '--prefill-chunk-size', str(args.prefill_chunk_size),
                    '--host-gib', str(args.host_gib),
                    *(['--no-graph'] if args.no_graph else []),
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
                        if args.disable_thinking:data['chat_template_kwargs']={'enable_thinking':False}
                        data.update(extra)
                        return urlopen(Request(base + "/v1/chat/completions", data=json.dumps(data).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=args.request_timeout)
                    def status():
                        with urlopen(base+'/v1/models/status',timeout=30) as response:
                            return json.load(response)
                    def unload(alias):
                        body=json.dumps(dict(model=alias,release_host=True)).encode()
                        with urlopen(Request(base+'/v1/models/unload',data=body,
                                headers={'Content-Type':'application/json'}),timeout=120) as response:
                            return json.load(response)
                    def allocated(state):
                        key={'cuda':'cuda_allocated_bytes','rocm':'hip_allocated_bytes','mps':'mps_allocated_bytes'}[state['device']]
                        return state[key]
                    trials = []
                    texts = {}
                    stable_allocations = {}
                    for alias in ("a", "b") * args.swap_cycles + ("a",):
                        start = time.perf_counter()
                        with post(alias) as response:
                            data = json.load(response)
                        text = data["choices"][0]["message"]["content"]
                        assert text, "empty completion cannot establish useful model-switching proof"
                        if alias in texts:
                            assert text == texts[alias], "model switch changed deterministic output"
                        texts[alias] = text
                        state=status()
                        if len(trials)>=4:
                            baseline=stable_allocations.setdefault(alias,allocated(state))
                            assert allocated(state)==baseline, 'live allocation grew after swap warmup'
                        trials.append({"model": alias, "wall_seconds": time.perf_counter() - start,
                                       "usage": data["usage"], "timing": data["dllm_timing"],
                                       "allocated_bytes":allocated(state)})
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
                    for alias, extra, expected_status in (("a", {"max_tokens": 10000}, 400), ("broken", {}, 500)):
                        try:
                            post(alias, **extra).close()
                            raise AssertionError("bad request unexpectedly succeeded")
                        except HTTPError as error:
                            assert error.code == expected_status, error.code
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
                    def same_model(_):
                        with post("a") as response:
                            return json.load(response)
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        batched = list(pool.map(same_model, range(4)))
                    for response in batched:
                        assert response["choices"][0]["message"]["content"] == texts["a"]
                    assert max(r["dllm_timing"]["max_batch_size"] for r in batched) == 4
                    benchmark = request_one(base + "/v1", "a", prompt, 64, 120, 0)
                    assert benchmark.error is None, benchmark.error
                    for alias in ('a','b'):unload(alias)
                    released=status()
                    assert released['host_cached_bytes']==0
                    assert all(m['residency']!='gpu' for m in released['models'])
                    with post('a') as response:
                        reloaded=json.load(response)
                    assert reloaded['choices'][0]['message']['content']==texts['a']
                    assert reloaded['dllm_timing']['source_tier']=='disk'
                    unload('a')
                    released_again=status()
                    assert released_again['host_cached_bytes']==0
                    assert allocated(released_again)==allocated(released), 'reload retained additional device allocations'
                    reports.append({"gpu_models": slots, "released_status":released,
                                    "released_again_status":released_again, "trials": trials, "sample_text": texts,
                                    "streaming_and_recovery_passed": True,
                                    "same_model_batch": [r["dllm_timing"] for r in batched],
                                    "stream_benchmark": vars(benchmark)})
                except Exception as error:
                    report['error']=repr(error)
                    report['failed_capacity']=slots
                    report['server_log']=str(log_path)
                    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
                    raise
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    report['passed']=True
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()

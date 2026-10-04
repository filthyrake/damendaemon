"""Rejected experiment: A2 HTTP comparison of Qwen2 precision policies.

Requires docs/evidence/gguf/experimental-qwen2-mixed.patch in an isolated
checkout. The policy regressed an additional numerical case and is NOT a
production runtime option. Retained only to reproduce the measured tradeoff.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, urlopen

import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer, handler_for


def main():
    p = argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('--output', required=True); args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('select the authorized A2 explicitly')
    from dllm.qwen2 import Executor
    if 'qwen2_precision' not in Executor.__init__.__code__.co_names:
        raise RuntimeError('apply the retained experimental precision patch in an isolated checkout first')
    torch.set_num_threads(4)
    runtime = BatchedRuntime({'standard':args.model, 'mixed':args.model}, context=512, batch_size=2, gpu_models=2, host_models=2)
    server = BoundedHTTPServer(('127.0.0.1', 0), handler_for(runtime))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    results = []
    try:
        policies = {}
        for policy in ('standard', 'mixed'):
            os.environ['DLLM_QWEN2_PRECISION'] = policy
            runtime.manage('load', policy)
            with runtime.runtime.session(policy) as (executor, _, _):
                assert executor.qwen2_precision == policy
                policies[policy] = dict(matrix_dtype=str(executor.compute_dtype), residual_dtype=str(executor.residual_dtype),
                                        rotary_precise=executor.rotary_precise, weight_format=executor.model.weight_format)
                base = executor.model.raw_tokens('Explain how the ocean influences coastal weather. Include evaporation, wind and rain. ')
        corpus = base * ((400+len(base)-1)//len(base))
        def request(policy, ids):
            body = dict(model=policy, prompt=ids, temperature=0, max_tokens=16)
            start = time.perf_counter()
            with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions', data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json'}), timeout=180) as response: result = json.load(response)
            assert result['dllm_timing']['source_tier'] == 'gpu'
            return result, time.perf_counter()-start
        for length, concurrency in ((16, 1), (129, 1), (400, 1), (129, 2)):
            expected = {}; rows = []
            def run(policy):
                start = time.perf_counter()
                if concurrency == 1: responses = [request(policy, corpus[:length])]
                else:
                    with ThreadPoolExecutor(concurrency) as pool:
                        responses = list(pool.map(lambda _: request(policy, corpus[:length]), range(concurrency)))
                for response, _ in responses:
                    pair = (response['choices'], response['usage'])
                    if policy not in expected: expected[policy] = pair
                    assert pair == expected[policy]
                    assert response['usage']['completion_tokens'] == 16
                return responses, time.perf_counter()-start
            for policy in ('standard', 'mixed'):
                for _ in range(3): run(policy)
            for trial in range(6):
                for policy in (('standard', 'mixed') if trial % 2 == 0 else ('mixed', 'standard')):
                    responses, elapsed = run(policy)
                    rows.append(dict(policy=policy, wall_seconds=elapsed,
                        responses=[dict(timing=r['dllm_timing'], usage=r['usage'], http_seconds=t) for r, t in responses]))
            if concurrency == 2:
                assert max(r['timing']['max_batch_size'] for row in rows for r in row['responses']) == 2
            result = dict(prompt_tokens=length, concurrency=concurrency, exact_cross_policy_responses=expected['standard'] == expected['mixed'],
                trials=rows, medians={p:statistics.median(r['wall_seconds'] for r in rows if r['policy'] == p) for p in ('standard', 'mixed')})
            results.append(result); print(json.dumps({'prompt_tokens':length, 'concurrency':concurrency,
                'medians':result['medians'], 'exact':result['exact_cross_policy_responses']}), flush=True)
    finally:
        for policy in runtime.model_ids(): runtime.manage('unload', policy, release_host=True)
        server.shutdown(); server.server_close(); thread.join(); runtime.close()
    Path(args.output).write_text(json.dumps(dict(passed=True, device=torch.cuda.get_device_name(), torch=torch.__version__,
        model=args.model, policies=policies, results=results), indent=2)+'\n')


if __name__ == '__main__': main()

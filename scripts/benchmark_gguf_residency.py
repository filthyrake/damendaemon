"""Matched warm HTTP import-path comparison across GPU residency policies."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.request import Request, urlopen

import torch
from damendaemon.scheduler import BatchedRuntime
from damendaemon.server import BoundedHTTPServer, handler_for


A2 = 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('hf'); p.add_argument('--output', required=True)
    p.add_argument('--quantization', choices=('q8_0', 'q4_0'), required=True); args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != A2: raise RuntimeError('select the authorized A2 explicitly')
    torch.set_num_threads(4)
    results = []
    for capacity in (2, 1):
        for order in (('gguf', 'hf'), ('hf', 'gguf')):
            runtime = BatchedRuntime({}, context=512, batch_size=2, gpu_models=capacity, host_models=2)
            server = BoundedHTTPServer(('127.0.0.1', 0), handler_for(runtime))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                for alias in order:
                    runtime.manage('register', alias, path=str(Path(getattr(args, alias)).resolve()),
                                   quantization='none' if alias == 'gguf' else args.quantization)
                def request(alias):
                    body = dict(model=alias, prompt='Explain binary search step by step:', temperature=0, max_tokens=16)
                    start = time.perf_counter()
                    with urlopen(Request(base+'/v1/completions', data=json.dumps(body).encode(),
                        headers={'Content-Type':'application/json'}), timeout=180) as response: value = json.load(response)
                    return value, time.perf_counter()-start
                expected = None; activation = []
                for alias in order:
                    value, elapsed = request(alias)
                    pair = (value['choices'], value['usage'])
                    if expected is None: expected = pair
                    assert pair == expected
                    activation.append(dict(model=alias, wall_seconds=elapsed, timing=value['damendaemon_timing']))
                rows = []
                for trial in range(10):
                    for alias in (order if trial % 2 == 0 else tuple(reversed(order))):
                        for _ in range(3): request(alias)
                        value, elapsed = request(alias)
                        assert (value['choices'], value['usage']) == expected
                        assert value['damendaemon_timing']['source_tier'] == 'gpu'
                        telemetry = subprocess.check_output(['nvidia-smi', '-i', A2,
                            '--query-gpu=pstate,clocks.current.graphics,clocks.current.memory,power.draw',
                            '--format=csv,noheader,nounits'], text=True).strip()
                        rows.append(dict(model=alias, wall_seconds=elapsed, usage=value['usage'],
                                         timing=value['damendaemon_timing'], telemetry=telemetry))
                result = dict(gpu_capacity=capacity, initial_load_order=order, activation=activation, exact_responses=True,
                              trials=rows, medians={alias:statistics.median(r['wall_seconds'] for r in rows if r['model'] == alias) for alias in order})
                results.append(result); print(json.dumps({'progress':result['medians'], 'capacity':capacity, 'order':order}), flush=True)
            finally:
                for alias in runtime.model_ids(): runtime.manage('unload', alias, release_host=True)
                server.shutdown(); server.server_close(); thread.join(); runtime.close()
                del server, runtime
                gc.collect(); torch.cuda.empty_cache()
    report = dict(passed=True, device=torch.cuda.get_device_name(), torch=torch.__version__, gguf=args.gguf, hf=args.hf,
                  quantization=args.quantization, scope='warm HTTP; three unmeasured warm requests before each sample; ten alternating rounds', results=results)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()

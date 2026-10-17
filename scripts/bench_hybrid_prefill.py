"""Warm scan-versus-FLA prefill comparison; correctness is a separate replay gate."""
import argparse
import gc
import json
import os
import time
import torch
from damendaemon.qwen2 import ModelWeights
from damendaemon.runtime import require_gpu, make_executor


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace')
    args = p.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    trace = torch.load(args.trace, weights_only=True, map_location='cpu')
    model = ModelWeights(trace['path'], trace['context'])
    events = [e for e in trace['trace'] if e['kind']=='prefill' and max(map(len,e['prompts'])) >= 64]
    timings = []
    for backend in ('scan','fla'):
        os.environ['DAMENDAEMON_HYBRID_PREFILL'] = backend
        executor = make_executor(model, max_batch=4)
        for trial in range(5):
            for event in events:
                torch.cuda.synchronize()
                start = time.perf_counter()
                executor.prefill_batch(event['prompts'], event['slots'])
                torch.cuda.synchronize()
                timings.append({'backend': backend, 'warmup': trial == 0,
                    'lengths': list(map(len,event['prompts'])), 'seconds': time.perf_counter()-start})
        del executor
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({'model': trace['path'], 'context': trace['context'], 'timings': timings}, indent=2))


if __name__ == '__main__':
    main()

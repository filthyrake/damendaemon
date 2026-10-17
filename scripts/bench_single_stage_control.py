"""Single-GPU placement control for the qualified distributed Q8 workload.

Weights and every generated token must match the distributed report. Hardware
placement differs deliberately; this is not a cross-runtime speedup claim.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time
import torch
from damendaemon.qwen2 import ModelWeights
from damendaemon.runtime import require_gpu, make_executor
from damendaemon.stage_identity import fingerprint


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(); p.add_argument('model'); p.add_argument('--trace', required=True)
    p.add_argument('--distributed-result', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.write_text(json.dumps(dict(passed=False, final_report=False))+'\n')
    try:
        require_gpu()
        torch.set_num_threads(4)
        reference = json.loads(Path(args.distributed_result).read_text())
        if not reference['passed'] or (reference['context'], reference['prompt_tokens'],
                reference['output_tokens'], reference['batch'], reference['quantization']) != (512, 255, 32, 1, 'q8_0'):
            raise ValueError('unsupported distributed comparison workload')
        expected = reference['groups'][0]['runs'][0]['tokens']
        if len(expected) != 32:raise ValueError('invalid expected continuation')
        trace = torch.load(args.trace, map_location='cpu', weights_only=True)
        if trace['context'] != 512 or trace.get('reference_quantization') != 'q8_0':
            raise ValueError('reference trace precision/context mismatch')
        prompt = next(prompt for event in trace['trace'] if event['kind'] == 'prefill'
                      for prompt in event['prompts'] if len(prompt) == 255)
        model = ModelWeights(args.model, 512, quantization='q8_0')
        identity = fingerprint(model).hex()
        if identity != reference['model_sha256']:raise ValueError('prepared weights differ from distributed run')
        executor = make_executor(model, use_graph=True, max_batch=1)
        groups = []
        for group in range(2):
            runs = []
            for iteration in range(8):
                torch.cuda.synchronize(); started = time.perf_counter()
                token = executor.prefill(prompt).argmax(-1).item()
                generated = [token]; first = time.perf_counter()
                for _ in range(31):
                    _, next_token = executor.decode(token)
                    token = next_token.item(); generated.append(token)
                torch.cuda.synchronize(); finished = time.perf_counter()
                if generated != expected:raise AssertionError('single-GPU output differs from distributed output')
                if iteration >= 2:
                    runs.append(dict(first_token_seconds=first-started, total_seconds=finished-started,
                                     decode_tokens_per_second=31/(finished-first), tokens=generated))
            groups.append(dict(group=group, runs=runs))
        runs = [row for group in groups for row in group['runs']]
        report = dict(passed=True, final_report=True, model=args.model, model_sha256=identity,
            quantization='q8_0', context=512, prompt_tokens=255, output_tokens=32, batch=1,
            hostname=socket.gethostname(), device=torch.cuda.get_device_name(), torch=str(torch.__version__),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), kernel_backend=executor.kernel_backend,
            scope='single-GPU placement control; active hardware differs from split; not HTTP or cross-runtime comparison',
            summaries={key:statistics.median(row[key] for row in runs)
                       for key in ('first_token_seconds','total_seconds','decode_tokens_per_second')}, groups=groups)
        output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report['summaries']))
    except BaseException as error:
        output.write_text(json.dumps(dict(passed=False, final_report=True, error=repr(error)))+'\n')
        raise


if __name__ == '__main__':main()

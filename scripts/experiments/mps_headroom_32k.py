#!/usr/bin/env python3
"""#4 headroom record: 2B hybrid at 32K context on Apple MPS.

Loads the pinned Qwen3.5-2B text backbone at context 32768 with the Metal
scan, chunk-prefills a full 32K-token prompt, decodes 16 tokens, and records
the memory envelope: live MPS allocations, the Metal recommended ceiling,
the process footprint, and macOS unified-memory pressure headroom. No
numerical claim (the 84-check 32K replay trace owns that).
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import json
import os
import resource
import subprocess
import sys
import time

os.environ['DAMENDAEMON_DEVICE'] = 'mps'

import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/damenknight/damendaemon/results/models/qwen35-2b'
CONTEXT = 32768
CHUNK = 2048
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.expanduser('~/damendaemon/results/mps-2b-32k-headroom.json')

from damendaemon.models import load as load_model
from damendaemon.hybrid import HybridExecutor

model = load_model(MODEL, CONTEXT)
executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
assert executor.metal_scan is not None, 'Metal scan not selected at 2B/32K'
print(f'[headroom] executor up: metal_scan={executor.metal_scan is not None}, '
      f'context={CONTEXT}')

# A real 32K prompt: repeat a varied paragraph to ~32K tokens via the tokenizer.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
paragraph = ('The lighthouse keeper logged every passing vessel: tonnage, '
             'flag, bearing, and the sea state at dawn. Records accumulated '
             'across decades, and the archive became the town\'s memory. ')
ids = tokenizer(paragraph, add_special_tokens=False)['input_ids']
prompt = (ids * (CONTEXT // len(ids) + 1))[:CONTEXT - 16]
print(f'[headroom] prompt tokens: {len(prompt)}')

torch.mps.empty_cache()
record = {'model': MODEL, 'context': CONTEXT, 'chunk': CHUNK,
          'metal_scan': True, 'torch': torch.__version__}


def snapshot(label):
    entry = {
        'mps_allocated_gib': round(torch.mps.current_allocated_memory() / 2**30, 3),
        'mps_recommended_max_gib': round(torch.mps.recommended_max_memory() / 2**30, 3),
        'process_rss_gib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3),
    }
    vm = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    # 'Pages free' is a poor headroom signal on macOS (the compressor and
    # speculative pages own most reclaimable memory); record it as a
    # footnote only, never as the headroom claim. The primary headroom
    # figure is the MPS recommended ceiling vs live allocation.
    for line in vm.splitlines():
        if line.startswith('Pages free'):
            free_pages = int(line.split(':')[1].strip().rstrip('.'))
            entry['macos_free_pages_gib'] = round(free_pages * 16384 / 2**30, 2)
    record[label] = entry
    print(f'[headroom] {label}: mps={entry["mps_allocated_gib"]} GiB / '
          f'{entry["mps_recommended_max_gib"]} GiB recommended, '
          f'rss={entry["process_rss_gib"]} GiB, '
          f'free={entry.get("macos_free_gib")} GiB')


snapshot('after_load')
start = time.perf_counter()
offset = 0
logits = None
while offset < len(prompt):
    final = offset + CHUNK >= len(prompt)
    logits = executor.prefill_chunk(prompt[offset:offset + CHUNK], 0, offset, final)
    offset += CHUNK
prefill_s = time.perf_counter() - start
record['prefill_32k_chunked_s'] = round(prefill_s, 3)
record['prefill_tokens_per_s'] = round(len(prompt) / prefill_s, 1)
snapshot('after_prefill')

start = time.perf_counter()
token = int(logits.argmax(-1)[0])
generated = [token]
for _ in range(15):
    _, nxt = executor.decode(token)
    token = int(nxt[0])
    generated.append(token)
record['decode_16_s'] = round(time.perf_counter() - start, 3)
record['decode_tokens_per_s'] = round(16 / record['decode_16_s'], 2)
snapshot('after_decode')
record['sample_generation'] = tokenizer.decode(generated[:16])

with open(OUT, 'w') as handle:
    json.dump(record, handle, indent=1)
print('[headroom] wrote', OUT)

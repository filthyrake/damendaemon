"""#605 gate 5: concurrency SLO cell at the B580's admitted scope (batch <= 4).

Phase S: four sequential single-stream requests (fairness baseline).
Phase C: four concurrent streaming requests; per-client wall times, the
max/median spread, and total completion are the SLO measurements.
Usage: xpu-concurrency.py <port>
"""
import json
import sys
import threading
import time
from http.client import HTTPConnection

if not __debug__:
    raise RuntimeError('gates on assertions; run without -O')

PROMPT = ("The blocked reduction of a quantized projection computes each output "
          "row from bounded tiles, bounding register pressure so the working set "
          "stays within on-chip registers. Homelab GPUs serve several users.")
TOKENS = 128


def emit(**event):
    print(json.dumps(event), flush=True)


def one(port, slot, results, temperature=0):
    payload = json.dumps({'model': 'a', 'prompt': PROMPT,
                          'max_tokens': TOKENS, 'temperature': temperature})
    conn = HTTPConnection('127.0.0.1', port, timeout=600)
    started = time.monotonic()
    conn.request('POST', '/v1/completions', payload,
                 {'Content-Type': 'application/json', 'Content-Length': str(len(payload))})
    body = json.loads(conn.getresponse().read())
    wall = time.monotonic() - started
    conn.close()
    results[slot] = dict(wall_seconds=round(wall, 3),
                         completion_tokens=body.get('usage', {}).get('completion_tokens'),
                         ok='choices' in body)


def main():
    port = int(sys.argv[1])
    results = {}
    threads = [threading.Thread(target=one, args=(port, slot, results))
               for slot in range(4)]
    started = time.monotonic()
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    total = time.monotonic() - started
    walls = sorted(results[slot]['wall_seconds'] for slot in range(4))
    emit(event='concurrent', clients=4, tokens_each=TOKENS,
         walls=walls, total_seconds=round(total, 3),
         spread_seconds=round(walls[-1] - walls[0], 3),
         all_ok=all(results[slot]['ok'] for slot in range(4)),
         aggregate_tokens_per_second=round(4 * TOKENS / total, 2))


if __name__ == '__main__':
    main()

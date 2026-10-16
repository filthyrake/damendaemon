"""Live HTTP sampling: filtering, seeded replay, concurrent requests and SSE."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
from dllm.bench import events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    results = {}
    with tempfile.TemporaryDirectory() as temporary:
        logpath = Path(temporary) / 'server.log'
        with logpath.open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'dllm', 'serve', '--port', '0',
                '--context', '512', '--batch-size', '4', '--model', 'a='+args.model], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and process.poll() is None:
                    lines = [x for x in logpath.read_text().splitlines() if x.startswith('dllm listening on ')]
                    if lines:
                        base = lines[-1].split()[-1]
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError(logpath.read_text())
                def request(return_result=False, **options):
                    body = {'model':'a', 'messages':[{'role':'user','content':'Invent a strange story about a robot gardener.'}],
                            'max_tokens':32, **options}
                    if 'max_completion_tokens' in options:
                        body.pop('max_tokens')
                    with urlopen(Request(base+'/v1/chat/completions', data=json.dumps(body).encode(),
                            headers={'Content-Type':'application/json'}), timeout=180) as response:
                        if options.get('stream'):
                            chunks = list(events(response, require_terminated=True))
                            assert chunks[-1] == '[DONE]'
                            return ''.join(json.loads(x)['choices'][0]['delta'].get('content','')
                                for x in chunks[:-1] if json.loads(x).get('choices'))
                        result = json.load(response)
                        return result if return_result else result['choices'][0]['message']['content']
                greedy = request(temperature=0)
                assert request(temperature=1, top_k=1, seed=8) == greedy
                assert len(greedy) >= 13
                needle = greedy[5:13]
                prefix = greedy[:greedy.index(needle)]
                stopped = request(stop=needle, return_result=True)
                assert stopped['choices'][0]['message']['content'] == prefix
                assert stopped['choices'][0]['finish_reason'] == 'stop'
                assert request(stop=['a stop that cannot occur 🦙', needle], stream=True) == prefix
                limited = request(max_completion_tokens=1, return_result=True)
                assert limited['usage']['completion_tokens'] == 1
                options = {'temperature':.8, 'top_p':.9, 'top_k':32, 'seed':123}
                baseline = request(**options)
                assert request(**options) == baseline
                assert request(**options, stream=True) == baseline
                with ThreadPoolExecutor(4) as pool:
                    concurrent = list(pool.map(lambda _: request(**options), range(4)))
                assert all(text == baseline for text in concurrent), concurrent
                alternatives = [request(**{**options, 'seed':seed}) for seed in (1,2,3)]
                assert len(set(alternatives + [baseline])) > 1
                results = {'passed':True, 'model':args.model, 'greedy':greedy,
                           'seeded':baseline, 'concurrent':concurrent, 'other_seeds':alternatives,
                           'stop_sequence':needle, 'stopped':stopped, 'limit_alias':limited}
            except Exception:
                print(logpath.read_text(), file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

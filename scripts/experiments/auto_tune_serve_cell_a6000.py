"""#674 cell 5 guarded cell: `damendaemon serve --auto-tune` end to end.

Boots the REAL serve with the opt-in calibration flag on the A6000: the
calibration must run inside startup (before the health endpoint binds),
emit the standalone harness's artifacts, and apply the winner to the
serving runtime; the calibrated serve must then answer a request from the
same pinned corpus. This is a machinery cell — which candidate wins the
0.8B sweep is reported, not gated (the known-ordering anchor was cell 3's
guarded cell; here the gate is the calibration→apply→serve pipeline).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
MODEL = '/home/damen/models/Qwen3.5-0.8B-r2fc0636'
PORT = 8153
CONTEXT = 2048
OUTPUT_TOKENS = 64
GPU_FLOOR_MIB = 20000
HOST_FLOOR_BYTES = 100 * 1024**3


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'gate requires the single-device A6000 guard, got {visible}')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}
    if free.get(A6000_UUID, 0) < GPU_FLOOR_MIB:
        raise ValueError(f'A6000 headroom below the {GPU_FLOOR_MIB} MiB floor')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None or available < HOST_FLOOR_BYTES:
        raise ValueError(f'host MemAvailable below the {HOST_FLOOR_BYTES} floor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tune-manifest', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    base = Path(args.output)
    base.parent.mkdir(parents=True, exist_ok=True)
    artifacts = base.parent / 'auto-tune-artifacts'
    if artifacts.exists() and any(artifacts.iterdir()):
        raise ValueError('auto-tune artifact directory must be fresh')
    served = base.with_suffix('.served.json')
    save(served, dict(version=1, models=dict(qwen08=dict(path=MODEL, quantization='q8_0'))))
    report = dict(completed=False, gate='#674 cell 5: --auto-tune pre-serve calibration',
                  model=MODEL, context=CONTEXT, artifacts=str(artifacts))
    save(args.output, report)
    serve_log = open(base.with_suffix('.serve.log'), 'w')
    server = subprocess.Popen(
        [sys.executable, '-m', 'damendaemon', 'serve',
         '--model-config', str(served), '--device', 'cuda', '--context', str(CONTEXT),
         '--port', str(PORT), '--auto-tune', str(args.tune_manifest),
         '--auto-tune-output', str(artifacts),
         '--gpu-models', '1', '--host-models', '1', '--host-gib', '20',
         '--queue-timeout', '120', '--queue-size', '8'],
        stdout=serve_log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 900
        healthy = False
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        healthy = True
                        break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(f'serve exited {server.returncode} during calibration/boot')
                time.sleep(5)
        if not healthy:
            raise RuntimeError('serve never became healthy (calibration or boot failed)')
        # Gate 1 — calibration artifacts: ranking with a winner, rows, refusals.
        names = sorted(p.name for p in artifacts.iterdir())
        required = {'ranking.json', 'ranking.model-config.json', 'refusals.json'}
        missing = required - set(names)
        if missing:
            raise RuntimeError(f'calibration artifacts incomplete: missing {sorted(missing)}')
        ranking = json.loads((artifacts / 'ranking.json').read_text())
        if ranking.get('winner') is None:
            raise RuntimeError('calibration produced no winner')
        rows = [name for name in names if name.startswith('row-candidate-')]
        if len(rows) < 2:
            raise RuntimeError(f'expected at least two candidate rows, got {rows}')
        applied_batch = ranking['rows_ranked'][0]['candidate']['max_batch'] if ranking['rows_ranked'] else None
        # The winner record carries the environment; the applied line is in
        # the serve log.
        log_text = base.with_suffix('.serve.log').read_text()
        if 'auto-tune applied' not in log_text:
            raise RuntimeError('the applied-calibration line is missing from the serve log')
        report['calibration'] = dict(
            winner=ranking['winner'], rows_ranked=ranking['rows_ranked'],
            rows_eliminated=ranking['rows_eliminated'],
            candidate_rows=rows, serve_log_has_applied=True)
        save(args.output, report)
        # Gate 2 — the calibrated serve answers from the same pinned corpus.
        from damendaemon.autotune_serve import pinned_corpus
        manifest = json.loads(args.tune_manifest.read_text())
        corpus = pinned_corpus(MODEL, manifest['corpus']['prompt_tokens'],
                               manifest['corpus']['prompt_sha256'])
        body = json.dumps(dict(model='qwen08', prompt=corpus, max_tokens=OUTPUT_TOKENS,
                               temperature=0, stream=True,
                               stream_options={'include_usage': True})).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        first_output = None
        usage = None
        error = None
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    payload = line[6:].strip()
                    if payload == b'[DONE]':
                        break
                    event = json.loads(payload)
                    if first_output is None and (event.get('choices') or [{}])[0].get('text'):
                        first_output = time.monotonic() - started
                    if event.get('usage'):
                        usage = event['usage']
        except Exception as failure:
            error = failure
        tokens = (usage or {}).get('completion_tokens')
        if error or tokens != OUTPUT_TOKENS:
            raise RuntimeError(f'calibrated serve failed the request: error={error} '
                                f'tokens={tokens}')
        report['request'] = dict(first_output_seconds=first_output,
                                 completion_tokens=tokens,
                                 total_seconds=time.monotonic() - started,
                                 error=None)
        report['completed'] = True
        save(args.output, report)
        print(f"[cell5] winner batch={applied_batch} rows={len(rows)} "
              f"request tokens={tokens} -> completed", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except Exception:
            server.kill()
        serve_log.close()
    if not report['completed']:
        raise SystemExit('cell failed a gate; see report')


if __name__ == '__main__':
    main()

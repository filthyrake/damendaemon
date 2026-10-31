"""#664 disconnect diagnostic — reproduce the zombie + OOM storm cheaply.

Fast bounded cells on the A6000 (short 4K-context prompts, ~40-160 s per
variant, vs the 6-minute 32K lifecycle cells) with full instrumentation:

- serve (context 4096, batch 1, chunk 127, queue 2) as in the #663 cells
  but scaled down;
- D1: stream client disconnects right after the first token (as cell 3);
- D2: raw-socket client disconnects mid-prefill (a TRUE mid-prefill
  close — urllib cannot do this, it waits for response headers);
- each variant then MONITORS for 120 s: /v1/models/status + scheduler
  status every 5 s, small probe requests at +5/+15/+30/+60/+90/+120 s
  (admit + first-output timing separates queue-admission release from
  slot release), serve-log OOM-warning capture, py-spy dumps of the
  serve process at +30 s and +90 s (thread stacks), and a final
  faulthandler SIGABRT stack dump of the serve child.

Diagnosis only — no fix, no retry of the #663 measurements. Guarded as
pre-declared on #664.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8132
CONTEXT = 4096
PROMPT_TOKENS = 3072
CHUNK = 127
MONITOR_SECONDS = 120
PROBE_DELAYS = (5, 15, 30, 60, 90, 119)


def get_json(path, timeout=10):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{path}', timeout=timeout) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def build_prompt_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    copies = 1
    while len(ids) < PROMPT_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    return ids[:PROMPT_TOKENS]


def probe(prompt_ids, max_tokens=8):
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids[:128], max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first = None
    tokens = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if first is None and (event.get('choices') or [{}])[0].get('text'):
                    first = time.monotonic() - started
                if event.get('usage'):
                    tokens = event['usage'].get('completion_tokens')
    except Exception as failure:
        error = str(failure)
    return dict(first_output_seconds=first, total_seconds=time.monotonic() - started,
                completion_tokens=tokens, error=error)


def stream_until_first_then_close(prompt_ids, max_tokens, hold_after_first):
    """Stream client: closes the connection hold_after_first seconds after the
    first content chunk (the #663 cell-3 shape)."""
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    record = dict(first_output_seconds=None, disconnected_seconds=None, output_chars=0,
                  error=None)
    try:
        response = urllib.request.urlopen(request, timeout=600)
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            if record['first_output_seconds'] is None and (event.get('choices') or [{}])[0].get('text'):
                record['first_output_seconds'] = time.monotonic() - started
            record['output_chars'] += len((event.get('choices') or [{}])[0].get('text', ''))
            if (record['first_output_seconds'] is not None
                    and time.monotonic() - started >= record['first_output_seconds'] + hold_after_first):
                record['disconnected_seconds'] = time.monotonic() - started
                response.close()
                break
    except Exception as failure:
        record['error'] = str(failure)
    return record


def raw_socket_disconnect_mid_prefill(prompt_ids, close_after):
    """Raw-socket client: send the full POST, never read, close after
    close_after seconds (a TRUE mid-prefill disconnect)."""
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=512,
                           temperature=0, stream=True)).encode()
    started = time.monotonic()
    record = dict(disconnected_seconds=None, error=None)
    try:
        with socket.create_connection(('127.0.0.1', PORT), timeout=30) as sock:
            head = (f'POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n'
                    f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n'
                    f'\r\n').encode()
            sock.sendall(head + body)
            time.sleep(close_after)
            record['disconnected_seconds'] = time.monotonic() - started
            sock.close()  # close with the response unread: RST on server write
    except Exception as failure:
        record['error'] = str(failure)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('d1-decode', 'd2-prefill', 'd3-status'), required=True)
    parser.add_argument('--close-after', type=float, default=3.0,
                        help='D2 only: seconds before the raw-socket close')
    parser.add_argument('--context', type=int, default=4096,
                        help='serve context (32768 reproduces the #663 shape)')
    parser.add_argument('--prompt-tokens', type=int, default=3072)
    parser.add_argument('--queue-size', type=int, default=2)
    parser.add_argument('--hold-after-first', type=float, default=2.0,
                        help='D1 only: seconds after first output before closing')
    args = parser.parse_args()
    global CONTEXT, PROMPT_TOKENS
    CONTEXT, PROMPT_TOKENS = args.context, args.prompt_tokens
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'diagnostic requires the single-device A6000 guard, got {visible}')
    import torch
    prompt = build_prompt_ids()
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, variant=args.variant, model=SNAPSHOT,
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS, chunk=CHUNK,
                  batch=1, quantization='q4_0', precision='compensated',
                  close_after=args.close_after, device=A6000_UUID,
                  torch=torch.__version__, rounds={}, statuses={}, serve_pyspy=[])
    save(args.output, report)
    environment = dict(os.environ)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    environment['PYTHONFAULTHANDLER'] = '1'
    serve_log_path = args.output.with_suffix('.serve.log')
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, '-m', 'scripts.experiments.hybrid_q4_a6000_probe', 'serve',
             '--model-config', str(manifest_path), '--device', 'cuda', '--context', str(CONTEXT),
             '--no-graph', '--prefill-chunk-size', str(CHUNK),
             '--batch-size', '1', '--port', str(PORT),
             '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '40', '--queue-timeout', '600',
             '--queue-size', str(args.queue_size)],
            env=environment, stdout=open(serve_log_path, 'w'), stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError('serve exited before becoming healthy')
                time.sleep(10)
        else:
            raise RuntimeError('serve did not become healthy within 900 seconds')
        warm = probe(prompt, 1)
        report['rounds']['warmup'] = warm
        report['statuses']['after_warmup'] = get_json('/v1/models/status')
        save(args.output, report)
        if warm.get('error'):
            raise RuntimeError(f"warmup failed: {warm['error']}")
        # The disconnect itself.
        if args.variant == 'd3-status':
            # A long decode runs while status is polled every 2 s: does the
            # observability endpoint 503 during an ACTIVE generation group?
            # The request holds 45 s past first output (~154 s prefill + 45 s
            # decode) then closes mid-decode; poll through the whole span.
            holder = {}

            def run_long():
                holder['value'] = stream_until_first_then_close(prompt, 512, hold_after_first=45.0)

            thread = threading.Thread(target=run_long, daemon=True)
            thread.start()
            active_status = {}
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline and 'value' not in holder:
                key = f'{round(time.monotonic() % 100000, 1)}'
                active_status[key] = get_json('/v1/models/status')
                report['statuses']['during_active'] = active_status
                save(args.output, report)
                time.sleep(2)
            report['statuses']['during_active'] = active_status
            disconnect = dict(note='long request held 45 s after first output, then closed',
                             **holder.get('value', {}))
            # The standard monitor follows the close.
            disconnected_at = time.monotonic()
        elif args.variant == 'd1-decode':
            disconnect = stream_until_first_then_close(prompt, 512, hold_after_first=args.hold_after_first)
        else:
            disconnect = raw_socket_disconnect_mid_prefill(prompt, args.close_after)
        report['rounds']['disconnect'] = disconnect
        save(args.output, report)
        disconnected_at = time.monotonic()

        # Monitoring window.
        def pyspy(tag):
            try:
                dump = subprocess.run(['/home/damen/3090-damendaemon/venv/bin/py-spy', 'dump',
                                       '--pid', str(server.pid), '--nonblocking'],
                                      capture_output=True, text=True, timeout=60)
                report['serve_pyspy'].append(dict(tag=tag, returncode=dump.returncode,
                                                  output=dump.stdout[-8000:] or dump.stderr[-2000:]))
            except Exception as failure:
                report['serve_pyspy'].append(dict(tag=tag, error=str(failure)))

        probes = {}
        statuses = {}
        next_probe = list(PROBE_DELAYS)
        pyspy_marks = (30, 90)
        while True:
            now = time.monotonic() - disconnected_at
            if next_probe and now >= next_probe[0]:
                delay = next_probe.pop(0)
                probes[f'+{delay}s'] = probe(prompt)
                save(args.output, report)
            elapsed_key = f'+{int(now)}s'
            statuses[elapsed_key] = get_json('/v1/models/status')
            report['statuses']['monitor'] = statuses
            report['rounds']['probes'] = probes
            if any(mark <= now < mark + 5 for mark in pyspy_marks):
                pyspy(f'+{int(now)}s')
            if now >= MONITOR_SECONDS:
                break
            time.sleep(5)
        # End of window: faulthandler stack dump of the serve child (SIGABRT
        # terminates it — this is the end of the diagnostic anyway).
        server.send_signal(subprocess.signal.SIGABRT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        report['statuses']['final'] = get_json('/v1/models/status')
        report['serve_exit_code'] = server.poll()
        report['completed'] = True
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == '__main__':
    main()

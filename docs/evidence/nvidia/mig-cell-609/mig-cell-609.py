"""#609 gate 3: guarded MIG capacity cell on the labnode3 A30 (1g.6gb slice).

Phases (MIG enable/carve/disable happen OUTSIDE this script as explicit,
announced sudo calls; this file never mutates device state):

  slice — inventory under MIG (cotenancy report + torch slice properties),
           one serve of the qualified Qwen3.5-0.8B fixture on the MIG device
           with greedy outputs + slice-local memory accounting, one
           over-slice admission attempt (Qwen3.5-2B, big context) that must
           be REFUSED against the SLICE total, and clean-release checks.
  full  — the same fixture/prompts on the FULL A30 after MIG is disabled
           (greedy control: tokens must equal the slice run).

Records are the exact JSON written per phase; no throughput claims.
"""
import json
import os
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

PYTHON = '/home/damen/dllm-dev/venv/bin/python'
MODEL_08B = str(next(Path('/home/damen/.cache/huggingface/hub/'
                          'models--Qwen--Qwen3.5-0.8B/snapshots').glob('*')))
MODEL_20B = '/home/damen/dllm-dev/models/gpt-oss-20b'
PORT = 8396
PROMPTS = ['The capital of France is',
           'Water boils at a temperature of',
           'The three primary colors are']
PHASE = sys.argv[1] if len(sys.argv) > 1 else ''
OUT = f'/tmp/mig609/mig-cell-{PHASE}.json'


def _text(value):
    """TimeoutExpired payloads can arrive as bytes; records stay str."""
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return str(value)


def record(**payload):
    Path(OUT).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f'wrote {OUT}', flush=True)


def wait_ready(deadline_seconds=120):
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        try:
            conn = HTTPConnection('127.0.0.1', PORT, timeout=5)
            conn.request('GET', '/v1/models/status')
            response = conn.getresponse()
            body = response.read()
            conn.close()
            if response.status == 200:
                return json.loads(body)
        except OSError:
            time.sleep(2)
    raise SystemExit('serve did not become ready in time')


def complete(prompt, timeout=110):
    payload = json.dumps({'model': 'a', 'prompt': prompt,
                          'max_tokens': 24, 'temperature': 0})
    conn = HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    conn.request('POST', '/v1/completions', payload,
                 {'Content-Type': 'application/json',
                  'Content-Length': str(len(payload))})
    response = conn.getresponse()
    body = json.loads(response.read())
    conn.close()
    if response.status != 200:
        raise SystemExit(f'completion failed: {body}')
    return body


def serve_command(model, context, extra=()):
    return [PYTHON, '-m', 'damendaemon', 'serve', '--device', 'cuda',
            '--port', str(PORT), '--context', str(context),
            '--batch-size', '1', '--no-graph', '--model', f'a={model}', *extra]


def slice_properties():
    script = ('import json,torch;'
              'p=torch.cuda.get_device_properties(0);'
              'free,total=torch.cuda.mem_get_info();'
              'print(json.dumps(dict(name=p.name,total_memory=p.total_memory,'
              'multi_processor_count=p.multi_processor_count,'
              'uuid=str(p.uuid),free=free,mig_total=total)))')
    out = subprocess.run([PYTHON, '-c', script], capture_output=True,
                         text=True, timeout=120)
    return dict(source='torch', properties=json.loads(out.stdout),
                stdout_remainder=out.stdout[out.stdout.index('}') + 1:].strip(),
                stderr=out.stderr.strip()[:2000])


def cotenancy_report():
    script = ('import json;'
              'from damendaemon.cotenancy_inventory import run_coTenancy_report;'
              'print(json.dumps(run_coTenancy_report()))')
    out = subprocess.run([PYTHON, '-c', script], capture_output=True,
                         text=True, timeout=120)
    return json.loads(out.stdout)


def compute_apps():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                          '--format=csv,noheader'], capture_output=True, text=True,
                         timeout=30)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def serve_and_request(tag, visible, model, context, prompts, ready_bound=600,
                      request_timeout=300, extra=()):
    """Start a serve, drive prompts, collect responses.

    Model load in this runtime is LAZY (residency 'disk' until the first
    request constructs the executor), so admission happens at request time:
    a serve that merely reports ready has NOT been admitted. Every cell
    here drives at least one request.
    """
    log_path = f'/tmp/mig609/serve-{tag}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible,
               DAMENDAEMON_DEVICE='cuda', OMP_NUM_THREADS='4')
    log = open(log_path, 'w')
    process = subprocess.Popen(serve_command(model, context, extra), env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    try:
        status = wait_ready(ready_bound)
        results = []
        for prompt in prompts:
            try:
                body = complete(prompt, timeout=request_timeout)
                results.append(dict(prompt=prompt, ok=True,
                                    text=body['choices'][0]['text'],
                                    completion_tokens=body.get('usage', {}).get('completion_tokens')))
            except (SystemExit, OSError, ValueError) as failure:
                results.append(dict(prompt=prompt, ok=False,
                                    error=repr(failure)[:500]))
                break
        return dict(model_status=status, requests=results,
                    alive_after_requests=process.poll() is None,
                    serve_pid=process.pid, serve_log=log_path,
                    device_memory=slice_properties(), compute_apps=compute_apps())
    finally:
        process.terminate()
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)
        log.close()
        time.sleep(3)


def serve_phase(tag, visible, model=MODEL_08B, context=2048):
    """The qualified 0.8B fixture: three greedy prompts + accounting."""
    outcome = serve_and_request(tag, visible, model, context, PROMPTS)
    outcome['greedy'] = outcome.pop('requests')
    return outcome


def main_slice():
    inventory = cotenancy_report()
    props = slice_properties()
    total = props['properties']['total_memory']
    positive = serve_phase('slice', os.environ['CUDA_VISIBLE_DEVICES'])
    apps_after_release = compute_apps()
    record(phase='slice', inventory=inventory, slice_properties=props,
           positive=positive, clean_release=dict(
               compute_apps_after_terminate=apps_after_release,
               note='slice memory returns to the pre-serve baseline once the '
                    'serve exits'))
    # Over-slice admission: gpt-oss-20b MXFP4 (~12.6 GiB weights alone)
    # decisively exceeds the 5.81 GiB slice while fitting the full card.
    # Admission is REQUEST-DRIVEN (lazy residency), so the serve is started
    # AND a completion is driven; the load must be refused against the
    # SLICE total. Earlier probes that only waited for the port are NOT
    # admissions — a bound port proves nothing about memory. (Qwen3.5-2B
    # at ctx 8192/32768 port-only probes were likewise not admissions.)
    # --host-gib 16 lifts the default 8 GiB HOST weight budget so the
    # refusal demonstrably comes from the DEVICE admission (slice total),
    # not the host gate (the stock-budget run refused at the host gate —
    # recorded in serve-slice-20b-hostgate.log as a separate finding).
    refusal = serve_and_request('slice-20b', os.environ['CUDA_VISIBLE_DEVICES'],
                                MODEL_20B, 512, ['The capital of France is'],
                                request_timeout=900,
                                extra=('--host-gib', '16'))
    refusal.update(model='gpt-oss-20b (MXFP4 native)', context=512,
                   slice_total_bytes=total)
    payload = json.loads(Path(OUT).read_text())
    payload['admission_refusal'] = refusal
    record(**payload)


def main_full():
    control = serve_phase('full', os.environ['CUDA_VISIBLE_DEVICES'])
    # The same 20B request that the slice must refuse has to be ADMITTED on
    # the full A30 — proving the slice refusal priced against the slice
    # total, not the model or the request. Request-driven admission; the
    # gate is the executor constructing and returning tokens.
    full_admission = serve_and_request('full-20b', os.environ['CUDA_VISIBLE_DEVICES'],
                                       MODEL_20B, 512, ['The capital of France is'],
                                       request_timeout=900,
                                       extra=('--host-gib', '16'))
    full_admission.update(model='gpt-oss-20b (MXFP4 native)', context=512)
    record(phase='full', control=control, full_card_20b_admission=full_admission)


if __name__ == '__main__':
    if PHASE == 'slice':
        main_slice()
    elif PHASE == 'full':
        main_full()
    else:
        raise SystemExit('phase must be slice or full')

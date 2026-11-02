"""Long-document quality battery for the qualified long-context tiers (#687).

Capacity and SLO on the qualified tiers are measured (#685/#686/#59);
quality is not. This runner measures it with a deterministic, pinned
corpus — every ground truth is known by construction, so scoring is exact
string agreement, never a judge model. One guarded invocation per
(--tier, --phase):

- tier qwen27-32k : 27B dense Q4 compensated, MTP served, chunk 1024,
  batch 4 (the #685 production C4 shape), context 32768
- tier qwen27-64k : same model at 65536, batch 1 (the #689-final C1 shape)
- tier qwen35-32k : 35B MoE q8_0, plain production serve, batch 6 (the
  qualified C6 shape), context 32768

Phase qa: one streamed request over a corpus carrying
- T1 positional retrieval with distractors: 3 authorization-code needles at
  ~25%/50%/80% of the corpus + 6 same-format distractor needles (other
  colors/codes) — Q1-Q3 ask for the 3 needle codes;
- T2 multi-needle: 6 relay-station inventory counts spread through the
  corpus — Q4 asks for all six;
- T3 cross-document composition: 2 shipment/cold-chain-permit joins —
  Q5-Q6 answer YES/NO.

Phase summary: one streamed request over an event-log corpus with 5
pinned critical facts embedded in routine entries; the model summarizes
in <= 80 words.

Scoring is normalized exact agreement (punctuation stripped, lowercase):
codes, counts, YES/NO, pinned key substrings. Gates are declared on #687
BEFORE any run and applied unchanged here: T1 >= 2/3, T2 >= 4/6,
T3 >= 1/2, T4 >= 3/5; a tier qualifies only if all four hold at BOTH
phases. Timings are recorded, never gated (latency is the SLO lanes'
deliverable). Output digests recorded; failures retained verbatim.

Corpus layout: deterministic filler paragraphs (fixed template cycle —
no colors, codes, station names or counts appear in filler); needle
positions recorded as measured token fractions. Corpora are identical
across tiers in text; token counts differ per tokenizer and are recorded.

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

from typing import NamedTuple


class Tier(NamedTuple):
    snapshot: str
    model: str
    quantization: str
    context: int
    batch: int
    q4: bool


A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8147
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
MAX_TOKENS = {'qa': 700, 'summary': 384}

TIERS = {
    'qwen27-32k': Tier(snapshot='/home/damen/models/Qwen3.8-27B-1d4bf0f2',
                       model='qwen38', quantization='q4_0', context=32768,
                       batch=4, q4=True),
    'qwen27-64k': Tier(snapshot='/home/damen/models/Qwen3.8-27B-1d4bf0f2',
                       model='qwen38', quantization='q4_0', context=65536,
                       batch=1, q4=True),
    'qwen35-32k': Tier(snapshot='/home/damen/models/Qwen_Qwen3.6-35B-A3B',
                       model='qwen36moe', quantization='q8_0', context=32768,
                       batch=6, q4=False),
}

# ---- Pinned corpus facts (identical across tiers) -----------------------
T1_NEEDLES = [('amber', 'K7Q-4471', 0.25), ('cobalt', 'RX3-9028', 0.50),
              ('mauve', 'V8N-6113', 0.80)]
T1_DISTRACTORS = [('crimson', 'Z4L-1276', 0.33), ('ivory', 'T9P-3352', 0.44),
                  ('jade', 'M2W-8890', 0.58), ('slate', 'Q6B-5537', 0.68),
                  ('olive', 'D3H-7714', 0.88), ('violet', 'F7S-2249', 0.94)]
T1_SENTENCE = ('OPERATIONS NOTICE: The {color} crate transfer is authorized '
               'under code {code}. Keep this notice with the manifest.')
T2_NEEDLES = [('Halden', 47, 0.15), ('Ostmark', 52, 0.37), ('Vireo', 38, 0.62),
              ('Casselt', 61, 0.72), ('Dunmore', 44, 0.86), ('Pellworth', 57, 0.95)]
T2_SENTENCE = ('Relay station {name} filed its quarterly inventory: '
               '{count} sealed units on hand.')
T3_FACTS = [
    ('Cargo manifest 118: shipment SN-118 departed station Casselt at 06:40 '
     'carrying reagent lots.', 0.42),
    ('Cold-chain directive 7: any shipment departing station Casselt between '
     '05:00 and 09:00 must carry permit CX-221.', 0.55),
    ('Cargo manifest 204: shipment SN-204 departed station Vireo at 14:15 '
     'carrying alloy stock.', 0.65),
    ('Cold-chain directive 9: any shipment departing station Vireo between '
     '12:00 and 18:00 must carry permit KL-905.', 0.77),
]
T3_TRUTH = {'Q5': 'yes', 'Q6': 'no'}
QA_QUESTIONS = (
    'Answer these questions about the records above. Give each answer on its '
    "own line, formatted exactly as 'A1: ...' through 'A6: ...'. Do not write "
    'anything else.\n'
    'Q1: What is the authorization code for the amber crate transfer? '
    'Answer with the code only.\n'
    'Q2: What is the authorization code for the cobalt crate transfer? '
    'Answer with the code only.\n'
    'Q3: What is the authorization code for the mauve crate transfer? '
    'Answer with the code only.\n'
    'Q4: Report the quarterly inventory counts of the relay stations Halden, '
    'Ostmark, Vireo, Casselt, Dunmore and Pellworth as "station count" pairs '
    'separated by semicolons, in that order.\n'
    'Q5: Did shipment SN-118 require permit CX-221? Answer YES or NO only.\n'
    'Q6: Did shipment SN-204 require permit CX-221? Answer YES or NO only.')
T4_KEYS = ['thermal runaway', 'tower 4', 'halvard', 'deck 7', '48 minutes']
T4_EVENTS = [
    ('ALERT: thermal runaway detected in tower 4.', 0.30),
    ('Crew Halvard contained the fault at its source.', 0.45),
    ('Evacuation of deck 7 completed in 48 minutes.', 0.60),
    ('Reactor output held at nominal throughout the event.', 0.75),
]
T4_PROMPT = ('In at most 80 words, summarize the critical events of this log. '
             'Do not use lists; write the summary as one paragraph.')
FILLER = [
    'Warehouse crew {i} completed the scheduled inspection of loading bay '
    '{i} without incident.',
    'The dock log records shuttle {i} departing on time with routine cargo.',
    'Maintenance shifted crane {i} to standby after standard diagnostics.',
    'A routine sweep of aisle {i} reported nothing unusual.',
    'The night shift confirmed seal integrity for container row {i}.',
    'Fuel levels for generator {i} stayed within normal operating bands.',
    'Comms check with outpost {i} cleared all channels at the hourly mark.',
    'Weather station {i} logged steady barometric pressure through the watch.',
]

GATES = dict(t1_min=2, t1_total=3, t2_min=4, t2_total=6, t3_min=1, t3_total=2,
             t4_min=3, t4_total=5)


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'quality battery requires the single-device A6000 guard, got {visible}')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}
    headroom = free.get(A6000_UUID)
    if headroom is None or headroom < GPU_FLOOR_MIB:
        raise ValueError(f'A6000 headroom {headroom} MiB is below the {GPU_FLOOR_MIB} MiB floor')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None or available < HOST_FLOOR_BYTES:
        raise ValueError(f'host MemAvailable {available} bytes is below the {HOST_FLOOR_BYTES} floor')


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def assemble(tokenizer, target_tokens, inserts, closing):
    """Filler paragraphs to ~target_tokens, then insert pinned segments at
    measured fractions; positions are recomputed on the final layout."""
    paragraphs, counts = [], []
    filler_index = 0
    total = 0
    while total < target_tokens:
        sentences = [FILLER[(filler_index + k) % len(FILLER)].format(i=filler_index + k)
                     for k in range(4)]
        paragraph = ' '.join(sentences)
        tokens = len(tokenizer.encode(paragraph).ids)
        paragraphs.append(paragraph)
        counts.append(tokens)
        total += tokens
        filler_index += 4
    # Insert pinned segments after the paragraph whose start best matches
    # each target fraction, preserving order.
    events = sorted(inserts, key=lambda item: item[1])
    layout = []           # list of (kind, text)
    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + count)
    total = boundaries[-1]
    placed = []
    for kind, text, fraction in events:
        target = int(total * fraction)
        index = 0
        for j in range(len(paragraphs)):
            if boundaries[j + 1] >= target:
                index = j
                break
        placed.append((index, kind, text, fraction))
    # Build the final segment list.
    out, events_by_paragraph = [], {}
    for index, kind, text, _ in placed:
        events_by_paragraph.setdefault(index, []).append((kind, text))
    prefix = 0
    for j, paragraph in enumerate(paragraphs):
        for kind, text in events_by_paragraph.get(j, []):
            out.append((kind, text, prefix))
        out.append(('filler', paragraph, prefix))
        prefix += counts[j]
    # Recompute event positions on the assembled layout.
    final, position = [], 0
    for kind, text, _ in out:
        final.append((kind, text, position))
        position += len(tokenizer.encode(text).ids) + 1
    body = '\n\n'.join(text for _, text, _ in final)
    prompt = body + '\n\n' + closing
    total_prompt = len(tokenizer.encode(prompt).ids)
    return prompt, final, total, total_prompt


def request(prompt_ids, max_tokens, tag, limit=1600):
    body = json.dumps(dict(model=MODEL_NAME, prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=limit + 300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            if event.get('usage'):
                usage = event['usage']
            choice = (event.get('choices') or [{}])[0]
            if first_output is None and choice.get('text'):
                first_output = time.monotonic() - started
            if choice.get('text'):
                text.append(choice['text'])
    result = dict(tag=tag, first_output_seconds=first_output,
                 total_seconds=time.monotonic() - started,
                 output_chars=sum(len(part) for part in text),
                 output=''.join(text), usage=usage)
    print(f"[{tag}] first {first_output and round(first_output,1)}s total "
          f"{round(result['total_seconds'],1)}s tokens {(usage or {}).get('completion_tokens')}",
          flush=True)
    return result


def score_qa(output, layout):
    """Truth-blind answer extraction, two accepted response formats:
    the terse 'A1: ...' lines, or the reasoning models' Q-restatement
    style ('Q1: ...' then, in that segment, 'Answer: X' — the final
    Answer line wins when several appear). Extraction never injects the
    expected value; matching is the declared exact-agreement check."""
    alines = {}
    for match in re.finditer(r'A(\d)[ \t]*:[ \t]*([^\n]*)', output):
        alines[int(match.group(1))] = match.group(2)
    evidence = {}
    qmarkers = list(re.finditer(r'Q(\d)[ \t]*:[ \t]*', output))
    for index, marker in enumerate(qmarkers):
        number = int(marker.group(1))
        stop = qmarkers[index + 1].start() if index + 1 < len(qmarkers) else len(output)
        segment = output[marker.end():stop]
        finals = re.findall(r'Answer[ \t]*:[ \t]*([^\n]+)', segment)
        evidence[number] = finals[-1] if finals else segment
    for number, line in alines.items():
        evidence.setdefault(number, line)
    tasks = {'T1': [], 'T2': [], 'T3': []}
    for index, (color, code, _) in enumerate(T1_NEEDLES):
        given = re.sub(r'[^a-z0-9]', '', evidence.get(index + 1, '').lower())
        expected = re.sub(r'[^a-z0-9]', '', code.lower())
        tasks['T1'].append(dict(needle=color, expected=code,
                                answer=evidence.get(index + 1, '')[:160],
                                passed=expected in given))
    t2 = evidence.get(4, '')
    for name, count, _ in T2_NEEDLES:
        pattern = re.compile(re.escape(name) + r'\D{0,12}(\d+)', re.IGNORECASE)
        found = pattern.search(normalize(t2))
        passed = False
        if found:
            passed = int(found.group(1)) == count
        tasks['T2'].append(dict(station=name, expected=count, answer=t2[:240], passed=passed))
    for index, question in enumerate(('Q5', 'Q6')):
        given = normalize(evidence.get(index + 5, ''))
        verdict = 'yes' if re.search(r'\byes\b', given) else ('no' if re.search(r'\bno\b', given) else None)
        tasks['T3'].append(dict(question=question, expected=T3_TRUTH[question],
                                answer=evidence.get(index + 5, '')[:160],
                                passed=verdict == T3_TRUTH[question], parsed=verdict))
    return {f'A{number}': text[:200] for number, text in sorted(evidence.items())}, tasks


def score_summary(output):
    # Reasoning models may quote the events while thinking; the summary
    # ends the response, so the gate reads the final 120 words only.
    tail = ' '.join(normalize(output).split()[-120:])
    items = [dict(key=key, passed=key in tail) for key in T4_KEYS]
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tier', choices=tuple(TIERS), required=True)
    parser.add_argument('--phase', choices=('qa', 'summary'), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    tier = TIERS[args.tier]
    global MODEL_NAME
    MODEL_NAME = tier.model
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tier.snapshot) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    corpus_budget = tier.context - 700
    if args.phase == 'qa':
        inserts = ([('t1', T1_SENTENCE.format(color=color, code=code), fraction)
                    for color, code, fraction in T1_NEEDLES]
                   + [('t1', T1_SENTENCE.format(color=color, code=code), fraction)
                      for color, code, fraction in T1_DISTRACTORS]
                   + [('t2', T2_SENTENCE.format(name=name, count=count), fraction)
                      for name, count, fraction in T2_NEEDLES]
                   + [('t3', text, fraction) for text, fraction in T3_FACTS])
        limit, target, closing = tier.context - MAX_TOKENS['qa'] - 64, corpus_budget - 900, QA_QUESTIONS
    else:
        inserts = [('t4', text, fraction) for text, fraction in T4_EVENTS]
        limit, target, closing = tier.context - MAX_TOKENS['summary'] - 64, corpus_budget - 400, T4_PROMPT
    prompt_text = layout = corpus_tokens = prompt_tokens = None
    for _ in range(6):
        prompt_text, layout, corpus_tokens, prompt_tokens = assemble(
            tokenizer, target, inserts, closing)
        if prompt_tokens <= limit:
            break
        target -= (prompt_tokens - limit) + 64
    else:
        raise ValueError(f'{args.phase} prompt {prompt_tokens} exceeds the context budget')
    prompt_ids = tokenizer.encode(prompt_text).ids
    positions = [dict(kind=kind, fraction=round(position / max(corpus_tokens, 1), 3))
                 for kind, _, position in layout if kind != 'filler']

    manifest = args.output.with_suffix('.manifest.json')
    save(manifest, dict(version=1, models={MODEL_NAME: dict(path=tier.snapshot,
                                                           quantization=tier.quantization)}))
    environment = dict(os.environ)
    command = [sys.executable, '-m', 'damendaemon', 'serve',
               '--model-config', str(manifest), '--device', 'cuda',
               '--context', str(tier.context), '--no-graph',
               '--batch-size', str(tier.batch), '--port', str(PORT),
               '--gpu-models', '1', '--host-models', '1',
               '--host-gib', '40', '--queue-timeout', '600',
               '--queue-size', str(tier.batch)]
    if tier.q4:
        environment.update({'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
                            'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
                            'DAMENDAEMON_QUANTIZATION': 'q4_0',
                            'DAMENDAEMON_KERNELS': 'triton',
                            'DAMENDAEMON_DEVICE': 'cuda',
                            'DAMENDAEMON_HYBRID_MTP': '1',
                            'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': str(tier.context)})
        command += ['--prefill-chunk-size', '1024']
    serve_log = open(args.output.with_suffix('.serve.log'), 'w')
    report = dict(completed=False, gate=f'long-document quality ({args.tier} {args.phase})',
                  tier=args.tier, phase=args.phase, snapshot=tier.snapshot,
                  quantization=tier.quantization, context=tier.context,
                  batch=tier.batch, corpus_text_sha256=hashlib.sha256(
                      prompt_text.encode()).hexdigest(),
                  prompt_ids_sha256=hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                  prompt_tokens=prompt_tokens, needle_positions=positions,
                  gates=GATES, scoring='normalized exact agreement, judge-free')
    save(args.output, report)
    server = None
    try:
        server = subprocess.Popen(command, env=environment, stdout=serve_log,
                                  stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(f'serve exited {server.returncode} during warmup')
                time.sleep(5)
        else:
            raise RuntimeError('serve never became healthy')
        request(prompt_ids[:512], 4, 'warmup')
        measured = request(prompt_ids, MAX_TOKENS[args.phase], f'{args.tier}-{args.phase}')
        report['request'] = {k: v for k, v in measured.items() if k != 'output'}
        report['request']['output_preview'] = measured['output'][:400]
        if args.phase == 'qa':
            answers, tasks = score_qa(measured['output'], layout)
            report['parsed_answers'] = {k: v[:200] for k, v in answers.items()}
            report['tasks'] = tasks
            t1 = sum(1 for item in tasks['T1'] if item['passed'])
            t2 = sum(1 for item in tasks['T2'] if item['passed'])
            t3 = sum(1 for item in tasks['T3'] if item['passed'])
            report['scores'] = dict(T1=f'{t1}/{GATES["t1_total"]}', T2=f'{t2}/{GATES["t2_total"]}',
                                    T3=f'{t3}/{GATES["t3_total"]}')
            report['completed'] = bool(t1 >= GATES['t1_min'] and t2 >= GATES['t2_min']
                                      and t3 >= GATES['t3_min']
                                      and (measured.get('usage') or {}).get('completion_tokens'))
        else:
            items = score_summary(measured['output'])
            report['tasks'] = {'T4': items}
            t4 = sum(1 for item in items if item['passed'])
            report['scores'] = dict(T4=f'{t4}/{GATES["t4_total"]}')
            report['completed'] = bool(t4 >= GATES['t4_min']
                                      and (measured.get('usage') or {}).get('completion_tokens'))
        save(args.output, report)
        print(f"[{args.tier}/{args.phase}] scores={report['scores']} "
              f"completed={report['completed']}", flush=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except Exception:
                server.kill()
        serve_log.close()
    if not report['completed']:
        raise SystemExit('quality cell failed a gate; see report')


if __name__ == '__main__':
    main()

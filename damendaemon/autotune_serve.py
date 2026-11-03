"""#674 cell 5: the opt-in `--auto-tune` pre-serve calibration.

Runs cell 3's bounded search INSIDE serve startup, before binding
admission: enumerate the manifest's declared candidates (cell 1's
production-policy enumerator), measure each against the manifest's
pinned workload, rank against the declared objective (cell 2), and emit
the same artifacts the standalone harness produces (cell 4) into the
declared output directory. The winner is then APPLIED to this serve's
own parameters (environment + batch/chunk/queue/graphs) before the
runtime is constructed — env vars are read at executor construction, so
applying them here is load-bearing and identical to launching serve
with the recommendation by hand.

Default remains fully manual: without the flag, nothing here runs.

The measurement is the qualified #671 C-series DIRECT-executor pattern
(sequential per-slot prefills, then continuously batched greedy decode),
not HTTP: no sockets are bound before calibration completes, and the
relative ordering across candidates is the decision surface. The rows
carry the same shapes as the standalone harness so one rank() serves
both.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from .autotune import (Measurement, declared_grid, enumerate_candidates,
                       parse_manifest, rank, winner_record)

C_SERIES_PARAGRAPH = ('The quick brown fox jumps over the lazy dog while the '
                      'repository records every bounded qualification cell '
                      'with evidence and pins. ')


class _EnvSession:
    """Candidate env for the duration of a measurement; ambient restored
    after (the winner's env is applied later, deliberately, by the caller)."""

    def __init__(self, overlay):
        self.overlay = overlay
        self.saved = {}

    def __enter__(self):
        for key, value in self.overlay.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def has_mtp_tensors(path):
    """Cheap checkpoint probe (safetensors index only; never a load)."""
    index = Path(path) / 'model.safetensors.index.json'
    if index.is_file():
        keys = json.loads(index.read_text()).get('weight_map', {})
        return any(name.startswith('mtp.') for name in keys)
    from .models import is_gguf
    if is_gguf(path):
        raise ValueError('GGUF MTP probing is not implemented; verify the '
                         'checkpoint by hand before declaring mtp in the '
                         'tune manifest')
    return False


def pinned_corpus(model_path, prompt_tokens, pinned_digest):
    """Digest-verified pinned corpus (the C-series token-ID convention)."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(model_path) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    ids = tokenizer.encode(C_SERIES_PARAGRAPH).ids
    if not ids:
        raise ValueError('prompt paragraph tokenized to zero ids')
    copies = 1
    while len(ids) < 2 * prompt_tokens:
        copies *= 2
        ids = tokenizer.encode(C_SERIES_PARAGRAPH * copies).ids
    ids = ids[:prompt_tokens]
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if digest != pinned_digest:
        raise ValueError('the pinned corpus digest does not match this '
                         'manifest; refusing to calibrate against a '
                         'different workload')
    return ids


def row_from_measurement(candidate, first_output, request_seconds,
                         tokens, round_seconds, metric, unmeasurable=None):
    """Shape a rank()-ready row from the direct measurement's signals."""
    aggregate = (sum(tokens) / round_seconds
                 if tokens and round_seconds > 0 else None)
    ordered = sorted(first_output) if first_output else []
    p95 = (ordered[max(0, int(round(len(ordered) * 0.95)) - 1)]
           if ordered else None)
    return dict(candidate=candidate.__dict__,
                metric_value=None if unmeasurable else
                (p95 if metric == 'p95_first_output_seconds' else aggregate),
                p95_first_output_seconds=p95,
                aggregate_output_tokens_per_second=aggregate,
                first_output_seconds=list(first_output or []),
                request_seconds=list(request_seconds or []),
                completion_tokens=list(tokens or []),
                errors=0, requests_measured=len(tokens or []),
                unmeasurable_reason=unmeasurable)


def direct_measure(candidate, manifest, model_path, corpus_ids):
    """The production measurement: one executor per candidate, driven the
    qualified C-series direct way (sequential per-slot whole prefills,
    then continuously batched greedy decode). A failing candidate returns
    an unmeasurable row (disclosed, ranked out) — one bad candidate never
    aborts the search."""
    requests_spec = manifest['requests']
    slots = min(requests_spec['concurrency'], requests_spec['count'])
    max_tokens = requests_spec['max_tokens']
    base = corpus_ids
    prompts = [base]
    for slot in range(1, slots):
        rotated = (base[slot:] + base[:slot])[:len(base)]
        if rotated == base:
            raise ValueError(f'slot {slot} rotation collided with slot 0')
        prompts.append(rotated)
    try:
        from . import models as models_module
        from .runtime import make_executor
        from .device import empty_cache, synchronize
        executor = None
        try:
            with _EnvSession(candidate.environment()):
                model = models_module.load(model_path, manifest['context'],
                                          quantization=candidate.quantization)
                executor = make_executor(model, candidate.graphs,
                                         candidate.max_batch,
                                         kernels=candidate.kernels)
                synchronize()
                # Warmup: flush JIT for the shapes before timing (C-series).
                logits = executor.prefill(prompts[0][:256], 0)
                token = int(logits[0].float().argmax())
                for _ in range(4):
                    _, nxt = executor.decode_batch([token], [0])
                    token = int(nxt[0])
                synchronize()
                started = time.perf_counter()
                # Sequential per-slot whole prefill; each slot's first token
                # comes from its own prefill logits (the C-series cells).
                first_tokens = []
                prefill_seconds = []
                for slot, prompt in enumerate(prompts):
                    slot_started = time.perf_counter()
                    logits = executor.prefill(prompt, slot)
                    synchronize()
                    prefill_seconds.append(time.perf_counter() - slot_started)
                    if not logits[0].float().isfinite().all():
                        raise ValueError(f'slot {slot} prefill logits not finite')
                    first_tokens.append(int(logits[0].float().argmax()))
                # Continuously batched greedy decode to max_tokens per slot.
                # Lockstep (no early EOS) keeps the work identical across
                # candidates — calibration compares configurations, not stop
                # behavior; disclosed.
                current = list(first_tokens)
                first_decode = None
                for step in range(max_tokens - 1):
                    _, nxt = executor.decode_batch(list(current),
                                                   list(range(slots)))
                    if first_decode is None:
                        synchronize()
                        first_decode = time.perf_counter() - started
                    current = [int(value) for value in nxt]
                synchronize()
                round_seconds = time.perf_counter() - started
                # First output per slot: the client-observed analog is the
                # first batched decode emission (prefills are sequential;
                # every slot's first generated token lands in that step).
                first_output = [first_decode] * slots
                request_seconds = [round_seconds] * slots
                return row_from_measurement(
                    candidate, first_output, request_seconds,
                    [max_tokens] * slots, round_seconds,
                    manifest['objective']['metric'])
        finally:
            if executor is not None:
                del executor
            try:
                synchronize()
                empty_cache()
            except Exception:  # noqa: BLE001 — cleanup never masks results
                pass
    except Exception as failure:  # noqa: BLE001 — disclosed, ranked out
        return row_from_measurement(candidate, None, None, None, 0,
                                    manifest['objective']['metric'],
                                    unmeasurable=f'{type(failure).__name__}: '
                                                 f'{failure}')


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=1))


def calibrate(manifest_path, output_dir, served_models, context, device,
              *, measure=None, enumerate_fn=None, corpus_fn=None):
    """The orchestration (pure — the measurement, enumeration and corpus
    are injectable seams so CPU tests cover it without a GPU, a checkpoint
    or a tokenizer). Returns (winner_record, applied) where applied
    carries this serve's tuned parameters; raises ValueError when the
    manifest cannot be honored, and RuntimeError when no candidate passes
    the gates (the serve refuses to start on a failed calibration —
    never silently falls back)."""
    manifest = parse_manifest(Path(manifest_path).read_bytes())
    model = manifest['model']
    if model['alias'] not in served_models:
        raise ValueError(f"tune manifest targets alias {model['alias']!r} "
                         'which is not served')
    resolved = served_models[model['alias']]
    if str(resolved) != model['path']:
        raise ValueError(f"served path {resolved} does not match the tune "
                         f"manifest path {model['path']}")
    if manifest['context'] != context:
        raise ValueError('the tune manifest context must equal the serve '
                        'context (the measured surface is the served one)')
    if manifest['hardware']['accelerator'] != device:
        raise ValueError('the tune manifest accelerator must equal the serve '
                        'device')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('auto-tune output directory must be empty or absent '
                         '(evidence is retained, never overwritten)')
    output.mkdir(parents=True, exist_ok=True)
    if enumerate_fn is None:
        from .models import config as model_config
        config = model_config(model['path'])
        enumerate_fn = enumerate_candidates
    else:
        config = None
    has_mtp = has_mtp_tensors(model['path'])
    declared = manifest['search'].get('values', {})
    declared_mtp_on = 'on' in declared.get('mtp', [])
    if declared_mtp_on != model['mtp']:
        raise ValueError('manifest model.mtp must match the declared mtp '
                         'search values')
    if model['mtp'] and not has_mtp:
        raise ValueError('the search declares MTP candidates but the '
                         'checkpoint carries no MTP tensors')
    grid = declared_grid(declared, model['quantization'])
    if enumerate_fn is None:
        enumerate_fn = enumerate_candidates
    admitted, refusals = enumerate_fn(config, accelerator=device,
                                     context=context, has_mtp=has_mtp,
                                     grid=grid)
    _write(output / 'refusals.json',
           [{'candidate': dict(candidate.__dict__), 'refusal': refusal}
            for candidate, refusal in refusals])
    if not admitted:
        raise RuntimeError('the declared search admits no candidates; '
                           f'refusals: {[r for _, r in refusals][:5]}')
    concurrency = manifest['requests']['concurrency']
    if any(candidate.max_batch < concurrency for candidate in admitted):
        raise ValueError('the workload concurrency must fit every admitted '
                         'candidate\'s batch (declare max_batch values that '
                         'all cover it)')
    if measure is None:
        measure = direct_measure
    if corpus_fn is None:
        corpus_fn = pinned_corpus
    corpus_ids = corpus_fn(model['path'],
                           manifest['corpus']['prompt_tokens'],
                           manifest['corpus']['prompt_sha256'])
    rows, raw_rows = [], []
    for index, candidate in enumerate(admitted):
        raw = measure(candidate, manifest, model['path'], corpus_ids)
        raw['candidate_index'] = index
        _write(output / f'row-candidate-{index}.json', raw)
        raw_rows.append(raw)
        rows.append(Measurement(
            candidate=candidate,
            metric_value=raw.get('metric_value'),
            first_output_seconds=raw.get('first_output_seconds') or [],
            request_seconds=raw.get('request_seconds') or [],
            completion_tokens=raw.get('completion_tokens') or [],
            errors=raw.get('errors') or 0,
            unmeasurable_reason=raw.get('unmeasurable_reason')))
    ranked, eliminated = rank(rows, manifest['objective'])
    report = dict(manifest_model=model, objective=manifest['objective'],
                  rows_ranked=[dict(candidate=row.candidate.__dict__,
                                    metric_value=row.metric_value)
                               for row in ranked],
                  rows_eliminated=[dict(candidate=row.candidate.__dict__,
                                        metric_value=row.metric_value,
                                        reasons=reasons)
                                   for row, reasons in eliminated],
                  winner=None)
    if not ranked:
        _write(output / 'ranking.json', report)
        raise RuntimeError('no candidate passed the declared gates; the '
                           'calibrated serve refuses to start')
    winner = ranked[0]
    record = winner_record(manifest, winner.candidate, winner.metric_value)
    report['winner'] = record
    _write(output / 'ranking.json', report)
    _write(output / 'ranking.model-config.json', record['model'])
    applied = dict(environment=record['environment'],
                   batch_size=winner.candidate.max_batch,
                   prefill_chunk_size=winner.candidate.prefill_chunk_size,
                   batch_wait_ms=winner.candidate.batch_wait_ms,
                   queue_size=winner.candidate.queue_size,
                   queue_timeout=winner.candidate.queue_timeout,
                   use_graph=winner.candidate.graphs)
    return record, applied

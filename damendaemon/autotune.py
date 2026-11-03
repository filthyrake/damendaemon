"""#674 auto-tune cells 1, 2 and 4: admission-aware configuration search.

Cell 1 — the tunable inventory and the admission-aware enumerator. The
enumerator NEVER re-implements policy: every refusal is produced by the
policy's own code path (`hybrid_q4_policy.validate`,
`hybrid_precision.select`, the MTP head's constructor gates), called in
serve order against a stand-in model descriptor carrying the same
config/quantization/context the policies read at serve time. The search
space is the PRODUCTION region: any ambient `DAMENDAEMON_EXPERIMENTAL_*`
opt-in makes the enumerator refuse loudly rather than search an
experiment view (isolated experiment views like the A6000 Q4 probe own
their own admission, by design).

Cell 2 — the versioned workload manifest: pinned model, digest-pinned
corpus, exact request counts, and the objective as a measurable predicate
(`best` is declared, not implied).

Cell 4 — the winner is emitted as a version-1 serve model-config manifest
(the exact shape `damendaemon serve --model-config` consumes) plus the
recommended environment and flags recorded in evidence for review.

Cell 3 (the guarded GPU search harness) lives in
scripts/experiments/auto_tune_search_a6000.py and consumes this module.
"""
from dataclasses import dataclass, field
from itertools import product
import hashlib
import json
import math
import os

MAX_MANIFEST_BYTES = 256 * 1024
MAX_DESCRIPTION_BYTES = 2048
OBJECTIVE_METRICS = ('p95_first_output_seconds', 'aggregate_output_tokens_per_second')
DIRECTIONS = {'p95_first_output_seconds': 'minimize',
              'aggregate_output_tokens_per_second': 'maximize'}
_GATE_KEYS = ('max_first_output_seconds', 'max_request_seconds',
              'exact_completion_tokens', 'zero_errors')
_EXPERIMENTAL_PREFIX = 'DAMENDAEMON_EXPERIMENTAL_'


@dataclass(frozen=True)
class Tunable:
    """One servable knob: its surface, its value set, and its gate."""
    name: str
    surface: str
    values: str
    gate: str


# The inventory is the whole admitted-tunability surface the search covers.
# Free-form numeric knobs (chunk size, batch-wait, queue sizing) are bounded
# per candidate by `admit` but are enumerated only when a manifest declares a
# value set for them — the objective decides whether they are worth grid space.
TUNABLES = (
    Tunable('quantization', 'serve manifest', 'none | q8_0 | q4_0',
            'hybrid_q4_policy.validate + packed-weight admission'),
    Tunable('precision', 'DAMENDAEMON_HYBRID_PRECISION',
            'float16 | float32 | compensated', 'hybrid_precision.select'),
    Tunable('kernels', 'DAMENDAEMON_KERNELS', 'auto | torch | triton',
            'hybrid_precision.select + executor kernel probe'),
    Tunable('graphs', '--no-graph', 'on | off',
            'hybrid_precision.select + MTP head constructor gates'),
    Tunable('kv_quant', 'DAMENDAEMON_HYBRID_KV', 'none | q8_0',
            'executor KV policy + MTP head gates'),
    Tunable('mtp', 'DAMENDAEMON_HYBRID_MTP (+ _DROP)', 'on | dropped',
            'MTP head constructor gates (checkpoint must carry the head)'),
    Tunable('mtp_spec_graph', 'DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH', '0 | 1',
            'the #691 opt-in; requires the MTP head, admits only eager-compatible graphs'),
    Tunable('prefill', 'DAMENDAEMON_HYBRID_PREFILL', 'scan | fla',
            'hybrid_precision.select (float32/compensated require scan)'),
    Tunable('max_batch', '--batch-size', '1..8 (policy may cap to 4)',
            'hybrid_precision.select + CLI choices'),
    Tunable('prefill_chunk_size', '--prefill-chunk-size', '0 (off) | declared positive set',
            'scheduler chunk budget validation'),
    Tunable('batch_wait_ms', '--batch-wait-ms', '0..10 ms', 'CLI bound'),
    Tunable('queue_size', '--queue-size', 'positive int', 'admission.validate_queue_limits'),
    Tunable('queue_timeout', '--queue-timeout', 'finite positive seconds',
            'admission.validate_queue_limits'),
)


@dataclass(frozen=True)
class Candidate:
    """One grid point over TUNABLES. The numerical serving knobs (chunk,
    batch-wait, queue) are carried for validation and emitted with the
    winner; a manifest decides which of them participate in the grid."""
    quantization: str = 'q8_0'
    precision: str = 'float16'
    kernels: str = 'triton'
    graphs: bool = True
    kv_quant: str = 'none'
    mtp: str = 'dropped'          # 'on' | 'dropped'
    mtp_spec_graph: bool = False
    prefill: str = 'scan'
    max_batch: int = 1
    prefill_chunk_size: int = 0
    batch_wait_ms: float = 0.0
    queue_size: int = 8
    queue_timeout: float = 120.0

    def environment(self):
        """The serve-process environment this candidate requires."""
        env = {'DAMENDAEMON_HYBRID_PRECISION': self.precision,
               'DAMENDAEMON_KERNELS': self.kernels,
               'DAMENDAEMON_HYBRID_KV': self.kv_quant,
               'DAMENDAEMON_HYBRID_PREFILL': self.prefill}
        if self.mtp == 'on':
            env['DAMENDAEMON_HYBRID_MTP'] = '1'
        else:
            env['DAMENDAEMON_HYBRID_MTP_DROP'] = '1'
        if self.mtp_spec_graph:
            env['DAMENDAEMON_HYBRID_MTP_SPEC_GRAPH'] = '1'
        return env

    def serve_flags(self):
        flags = [f'--batch-size', str(self.max_batch)]
        if not self.graphs:
            flags.append('--no-graph')
        if self.prefill_chunk_size:
            flags += ['--prefill-chunk-size', str(self.prefill_chunk_size)]
        if self.batch_wait_ms:
            flags += ['--batch-wait-ms', str(self.batch_wait_ms)]
        return flags


_DEFAULT_GRID = {
    'quantization': ('none', 'q8_0', 'q4_0'),
    'precision': ('float16', 'float32', 'compensated'),
    'kernels': ('auto', 'torch', 'triton'),
    'graphs': (True, False),
    'kv_quant': ('none', 'q8_0'),
    'mtp': ('dropped', 'on'),
    'mtp_spec_graph': (False, True),
    'prefill': ('scan', 'fla'),
    'max_batch': (1, 2, 4, 8),
}


@dataclass(frozen=True)
class _AdmissionModel:
    """The stand-in the policies read: the same fields HybridExecutor's
    admission path consumes, sourced from models.config without weights."""
    config: dict
    quantization: str
    max_context: int
    vision_weights: None = None


class _EnvSession:
    """Set the candidate's environment and restore the ambient one after —
    even on refusal. The tuner never writes an experimental opt-in."""

    def __init__(self, overlay):
        self.overlay = overlay
        self.saved = {}

    def __enter__(self):
        for key, value in self.overlay.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value
        for key in list(os.environ):
            if key.startswith(_EXPERIMENTAL_PREFIX):
                raise ValueError('auto-tune requires the production policy surface; '
                                 f'unset {key} (isolated experiment views own their own admission)')
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def _admit_numeric(candidate, context):
    """CLI-bound numeric knobs, validated before any policy call (mirrors
    the serve CLI's own bounds so refusals happen with recognizable text)."""
    if type(candidate.max_batch) is not int or not 1 <= candidate.max_batch <= 8:
        raise ValueError('batch size must be 1..8')
    if (type(candidate.batch_wait_ms) not in (int, float)
            or not math.isfinite(candidate.batch_wait_ms) or not 0 <= candidate.batch_wait_ms <= 10):
        raise ValueError('idle prompt coalescing delay must be 0..10 ms')
    if type(candidate.prefill_chunk_size) is not int or candidate.prefill_chunk_size < 0:
        raise ValueError('prefill chunk size must be a nonnegative integer')
    if type(context) is not int or context < 1:
        raise ValueError('context must be a positive integer')


def admit(candidate, config, *, accelerator, context, has_mtp,
          quantized_kernels_import=True):
    """Return (admitted, refusal). Refusals carry the policy's own error
    text — produced by the policy functions, never paraphrased here.
    `quantized_kernels_import` is the host capability the executor's own
    admission probe measures (packed eager-CUDA torch kernels qualify when
    the Triton projection kernels import); the search harness validates it
    once on the target host before enumerating."""
    from .hybrid_q4_policy import validate as validate_q4
    from .hybrid_precision import select as select_precision
    model = _AdmissionModel(config=config, quantization=candidate.quantization,
                            max_context=context)
    try:
        _admit_numeric(candidate, context)
        with _EnvSession(dict(candidate.environment(), DAMENDAEMON_DEVICE=accelerator)):
            # Serve admission order: quantization policy first (the Q4
            # surface refuses without its explicit opt-in — production
            # admission, by design), then precision, then the executor's
            # kernel/prefill gates, then the MTP gates.
            if candidate.quantization == 'q4_0':
                validate_q4(config, context)
            select_precision(model, candidate.graphs, candidate.max_batch, candidate.kernels)
            if candidate.prefill == 'fla' and candidate.kernels == 'torch':
                raise ValueError('FLA prefill requires the CUDA Triton backend')
            if (candidate.quantization in ('q8_0', 'q4_0')
                    and candidate.kernels == 'torch' and accelerator == 'cuda'
                    and not quantized_kernels_import):
                raise ValueError('hybrid packed weights on eager CUDA require '
                                 'the Triton projection kernels')
            if candidate.mtp == 'on':
                # Constructor order: the graphs refusal fires before the
                # checkpoint/kv gates (HybridExecutor's own sequence).
                if candidate.graphs and not candidate.mtp_spec_graph:
                    raise ValueError('the MTP draft head requires eager decode '
                                     '(no CUDA graphs) without the spec-graph opt-in')
                if not has_mtp:
                    raise ValueError('the MTP draft head requires a checkpoint '
                                     'that carries MTP tensors')
                if candidate.kv_quant != 'none':
                    raise ValueError('the MTP draft head is not qualified with quantized KV')
            if candidate.mtp_spec_graph:
                # Search-space hygiene, not policy: the #691 capture block
                # silently no-ops without the head, graphs, or a float32
                # compute — the tuner refuses inert combinations instead of
                # enumerating indistinguishable candidates.
                if (candidate.mtp != 'on' or not candidate.graphs
                        or candidate.precision not in ('float32', 'compensated')):
                    raise ValueError('the spec-graph opt-in requires the MTP head, '
                                     'CUDA graphs and a float32 compute '
                                     '(float32 or compensated precision)')
    except ValueError as refusal:
        return False, str(refusal)
    return True, None


COLLAPSED_DEFAULTS = {
    # The serve's own defaults: the axes a manifest does not declare
    # collapse to these (the search sweeps only what the manifest says).
    'precision': 'float16', 'kernels': 'auto', 'graphs': True, 'kv_quant': 'none',
    'mtp': 'dropped', 'mtp_spec_graph': False, 'prefill': 'scan', 'max_batch': 4,
}


def declared_grid(declared, quantization):
    """The manifest fully owns its search: every axis it does not declare
    collapses to the serve's own default (never to a sweep)."""
    if not isinstance(declared, dict):
        raise ValueError('declared values must be an object')
    allowed = set(_DEFAULT_GRID) | {'prefill_chunk_size'}
    unknown = set(declared) - allowed
    if unknown:
        raise ValueError(f'unknown tunables in the declared value set: {sorted(unknown)}')
    if quantization not in ('none', 'q8_0', 'q4_0'):
        raise ValueError('the model quantization must be none, q8_0 or q4_0')
    grid = {axis: (value,) for axis, value in COLLAPSED_DEFAULTS.items()}
    grid['quantization'] = (quantization,)
    for axis, values in declared.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'declared values for {axis} must be a nonempty list')
        grid[axis] = tuple(values)
    return grid


def enumerate_candidates(config, *, accelerator, context, has_mtp,
                         grid=None, extra_values=None,
                         quantized_kernels_import=True):
    """Cell 1: enumerate the admitted region. Returns
    (admitted, refusals) with every refusal's policy text retained so the
    search's exclusions are reviewable. `grid` REPLACES the default value
    sets (a manifest owns its search through `declared_grid`);
    `extra_values` overlays onto the defaults for ad-hoc probes."""
    if grid is not None:
        values = dict(grid)
    else:
        values = dict(_DEFAULT_GRID)
    if extra_values:
        unknown = set(extra_values) - set(values)
        if unknown:
            raise ValueError(f'unknown tunables in the declared value set: {sorted(unknown)}')
        values.update(extra_values)
    grid_names = ('quantization', 'precision', 'kernels', 'graphs', 'kv_quant',
                  'mtp', 'mtp_spec_graph', 'prefill', 'max_batch',
                  'prefill_chunk_size')
    names = tuple(name for name in grid_names if name in values)
    admitted, refusals = [], []
    seen_refusals = set()
    for combo in product(*(values[name] for name in names)):
        knobs = dict(zip(names, combo))
        candidate = Candidate(
            quantization=knobs.get('quantization', 'q8_0'),
            precision=knobs.get('precision', 'float16'),
            kernels=knobs.get('kernels', 'triton'),
            graphs=knobs.get('graphs', True),
            kv_quant=knobs.get('kv_quant', 'none'),
            mtp=knobs.get('mtp', 'dropped'),
            mtp_spec_graph=knobs.get('mtp_spec_graph', False),
            prefill=knobs.get('prefill', 'scan'),
            max_batch=knobs.get('max_batch', 1),
            prefill_chunk_size=knobs.get('prefill_chunk_size', 0))
        ok, refusal = admit(candidate, config, accelerator=accelerator,
                            context=context, has_mtp=has_mtp,
                            quantized_kernels_import=quantized_kernels_import)
        if ok:
            admitted.append(candidate)
        elif (candidate, refusal) not in seen_refusals:
            seen_refusals.add((candidate, refusal))
            refusals.append((candidate, refusal))
    return admitted, refusals


# ── Cell 2: the workload manifest and the objective predicate ──────────

def _validate_digest(value, field_name):
    if (type(value) is not str or len(value) != 64
            or not all(c in '0123456789abcdef' for c in value)):
        raise ValueError(f'{field_name} must be a lowercase sha256 hex digest')


def parse_manifest(raw, *, max_bytes=MAX_MANIFEST_BYTES):
    """Validate a tune manifest (bytes or str) and return it as a dict.
    Mirrors the fixed-workload discipline: exact integer counts, bounded
    bytes, digest-pinned corpus, a declared measurable objective."""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
        if len(raw) > max_bytes:
            raise ValueError('tune manifest exceeds the byte bound')
        raw = raw.decode('utf-8')
    if not isinstance(raw, str):
        raise ValueError('tune manifest must be text or bytes')
    if len(raw.encode()) > max_bytes:
        raise ValueError('tune manifest exceeds the byte bound')
    try:
        manifest = json.loads(raw, parse_int=lambda v: _strict_int(v),
                              parse_float=lambda v: _strict_float(v))
    except json.JSONDecodeError as error:
        raise ValueError(f'tune manifest is not valid JSON: {error.msg}') from error

    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('tune manifest schema_version must be 1')
    model = manifest.get('model')
    if (not isinstance(model, dict)
            or not isinstance(model.get('alias'), str) or not model['alias']
            or not isinstance(model.get('path'), str) or not model['path']
            or model.get('quantization') not in ('none', 'q8_0', 'q4_0')
            or type(model.get('mtp')) is not bool):
        raise ValueError('model requires alias, path, quantization in '
                         'none|q8_0|q4_0, and a boolean mtp field')
    hardware = manifest.get('hardware')
    if (not isinstance(hardware, dict)
            or hardware.get('accelerator') not in ('cuda', 'rocm', 'mps', 'xpu')):
        raise ValueError('hardware requires an accelerator in cuda|rocm|mps|xpu')
    context = manifest.get('context')
    if type(context) is not int or context < 1:
        raise ValueError('context must be a positive integer')
    corpus = manifest.get('corpus')
    if not isinstance(corpus, dict):
        raise ValueError('corpus must be an object')
    _validate_digest(corpus.get('prompt_sha256'), 'corpus.prompt_sha256')
    if type(corpus.get('prompt_tokens')) is not int or corpus['prompt_tokens'] < 1:
        raise ValueError('corpus.prompt_tokens must be a positive integer')
    description = corpus.get('description', '')
    if not isinstance(description, str) or len(description.encode()) > MAX_DESCRIPTION_BYTES:
        raise ValueError('corpus.description must be bounded text')
    requests = manifest.get('requests')
    if (not isinstance(requests, dict)
            or type(requests.get('count')) is not int or requests['count'] < 1
            or type(requests.get('max_tokens')) is not int or requests['max_tokens'] < 1
            or type(requests.get('concurrency')) is not int
            or not 1 <= requests['concurrency'] <= requests['count']):
        raise ValueError('requests requires positive count/max_tokens and '
                         'concurrency within 1..count')
    objective = manifest.get('objective')
    if not isinstance(objective, dict):
        raise ValueError('objective must be an object')
    metric = objective.get('metric')
    if metric not in OBJECTIVE_METRICS:
        raise ValueError(f'objective.metric must be one of {OBJECTIVE_METRICS}')
    if objective.get('direction') != DIRECTIONS[metric]:
        raise ValueError(f'objective.direction must be {DIRECTIONS[metric]!r} '
                         f'for {metric}')
    gates = objective.get('gates')
    if not isinstance(gates, dict) or set(gates) - set(_GATE_KEYS):
        raise ValueError(f'objective.gates may set only {list(_GATE_KEYS)}')
    for key in ('max_first_output_seconds', 'max_request_seconds'):
        if key in gates and (type(gates[key]) not in (int, float)
                             or not math.isfinite(gates[key]) or gates[key] <= 0):
            raise ValueError(f'{key} must be finite and positive')
    if 'exact_completion_tokens' in gates and (type(gates['exact_completion_tokens']) is not int
                                               or gates['exact_completion_tokens'] < 1):
        raise ValueError('exact_completion_tokens must be a positive integer')
    if 'zero_errors' in gates and type(gates['zero_errors']) is not bool:
        raise ValueError('zero_errors must be a boolean')
    search = manifest.get('search')
    if not isinstance(search, dict):
        raise ValueError('search must be an object')
    if (type(search.get('rounds')) is not int or search['rounds'] < 1
            or type(search.get('timeout_seconds')) is not int
            or not 1 <= search['timeout_seconds'] <= 1800):
        raise ValueError('search requires rounds >= 1 and timeout_seconds in 1..1800')
    declared = search.get('values', {})
    if not isinstance(declared, dict):
        raise ValueError('search.values must be an object')
    return manifest


def _strict_int(value):
    if not value.isascii() or not value.isdecimal():
        raise ValueError('non-integer where an integer was required')
    return int(value)


def _strict_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('non-finite number in the tune manifest')
    return number


def corpus_digest(prompt):
    """The pinned corpus digest recorded in the manifest."""
    return hashlib.sha256(prompt.encode()).hexdigest()


@dataclass
class Measurement:
    """One candidate's measured row (cell 3 fills these in)."""
    candidate: Candidate
    metric_value: float | None = None
    first_output_seconds: list = field(default_factory=list)
    request_seconds: list = field(default_factory=list)
    completion_tokens: list = field(default_factory=list)
    errors: int = 0
    unmeasurable_reason: str | None = None


def evaluate_gates(row, gates):
    """Apply the manifest's gates to one measurement; returns reasons."""
    reasons = []
    if row.unmeasurable_reason is not None:
        reasons.append(row.unmeasurable_reason)
        return reasons
    if row.errors:
        if gates.get('zero_errors'):
            reasons.append(f'{row.errors} errors')
        else:
            reasons.append(f'{row.errors} errors (disclosed, not gated)')
    if 'exact_completion_tokens' in gates:
        if not row.completion_tokens or any(
                tokens != gates['exact_completion_tokens'] for tokens in row.completion_tokens):
            reasons.append('completion tokens did not match the exact gate')
    if 'max_first_output_seconds' in gates and row.first_output_seconds:
        worst = max(row.first_output_seconds)
        if worst > gates['max_first_output_seconds']:
            reasons.append(f'first output {worst:.1f}s exceeded '
                           f"{gates['max_first_output_seconds']}s")
    if 'max_request_seconds' in gates and row.request_seconds:
        worst = max(row.request_seconds)
        if worst > gates['max_request_seconds']:
            reasons.append(f'request {worst:.1f}s exceeded '
                           f"{gates['max_request_seconds']}s")
    return reasons


def rank(rows, objective):
    """Rank measured candidates against the declared objective. Gate
    failures rank below gate passers; unmeasurable rows are disclosed,
    never guessed. Returns (ranked, eliminated)."""
    metric = objective['metric']
    gates = objective.get('gates', {})
    ranked, eliminated = [], []
    for row in rows:
        reasons = evaluate_gates(row, gates)
        if row.unmeasurable_reason is not None or row.metric_value is None:
            eliminated.append((row, reasons or ['unmeasurable: no metric value']))
        elif reasons:
            eliminated.append((row, reasons))
        else:
            ranked.append(row)
    ranked.sort(key=lambda row: row.metric_value,
                reverse=objective['direction'] == 'maximize')
    return ranked, eliminated


# ── Cell 4: the winner's pinned, serve-consumable output ────────────────

def serve_manifest(alias, path, quantization):
    """The version-1 model-config manifest `damendaemon serve --model-config`
    consumes as-is (the same shape the serve harnesses emit)."""
    return {'version': 1, 'models': {alias: {'path': path, 'quantization': quantization}}}


def winner_record(manifest, candidate, metric_value, gates_reasons=()):
    """The reviewable recommendation: the manifest the serve consumes, the
    environment/flags to launch it with, and the objective's verdict."""
    model = manifest['model']
    return {
        'model': serve_manifest(model['alias'], model['path'], candidate.quantization),
        'environment': candidate.environment(),
        'serve_flags': candidate.serve_flags(),
        'objective': manifest['objective'],
        'metric_value': metric_value,
        'gate_reasons': list(gates_reasons),
    }

"""HTTP serving adapter over an exclusively owned three-rank chain pipeline.

Mirrors the two-rank HostServingPair/HostDistributedRuntime shape: the pair
adapter exposes the surface the proven DistributedRuntime scheduler drives
(prefill/decode/release/heartbeat/stop over ChainPipeline), the runtime
subclass reports truthful chain metadata, and handler_for_chain adds chain
/health. Ranks 1..head run ChainStageWorker.run_forever; only rank 0
exposes HTTP. Model identity agreement across ranks is the caller's
startup gate (chain_server_cli verifies it before serving).
"""
from types import SimpleNamespace

from .chain_pipeline import ChainPipeline
from .distributed_serving import DistributedRuntime, STREAM_BUFFER_EVENTS


class ChainServingPair:
    """Transfer a ready chain pipeline and matching models to HTTP.

    The scheduler (DistributedRuntime) is the only inference owner; close may
    interrupt it from another thread. The primary alias serves first;
    `models` maps every admitted alias (identity + matching model surface),
    and `switch_model` swaps the resident shard table when the scheduler
    defers a request for a different alias.
    """

    rank = 0
    packed_prefill = False

    def __init__(self, pipeline, model, *, alias='model', identity, stage_layers,
                 chunk_size=128, models=None, identities=None,
                 stage_quantizations=None, stage_weight_bytes=None):
        if not isinstance(pipeline, ChainPipeline):
            raise ValueError('chain serving needs a ChainPipeline')
        if (pipeline.failed or pipeline.closed or any(pipeline.state.lengths)
                or getattr(pipeline, '_completion_owner', None) is not None):
            raise ValueError('chain serving requires an unclaimed ready empty pipeline')
        if (not isinstance(alias, str) or not 1 <= len(alias) <= 128
                or not isinstance(identity, (bytes, bytearray)) or len(identity) != 32):
            raise ValueError('invalid chain serving alias or model identity')
        if (not isinstance(stage_layers, (list, tuple)) or not stage_layers
                or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                       or any(type(bound) is not int or bound < 0 for bound in pair)
                       or pair[0] >= pair[1] for pair in stage_layers)
                or [pair[0] for pair in stage_layers][0] != 0
                or any(stage_layers[i][1] != stage_layers[i + 1][0]
                       for i in range(len(stage_layers) - 1))):
            raise ValueError('chain serving needs contiguous stage layer bounds')
        if stage_quantizations is not None and (not isinstance(stage_quantizations, (list, tuple))
                or len(stage_quantizations) != len(stage_layers)
                or any(not isinstance(value, str) or value not in ('none', 'q4_0', 'q8_0')
                       for value in stage_quantizations)):
            raise ValueError('chain serving per-rank formats must be admitted formats for every rank')
        if stage_weight_bytes is not None and (not isinstance(stage_weight_bytes, (list, tuple))
                or len(stage_weight_bytes) != len(stage_layers)
                or any(type(value) is not int or value < 0 for value in stage_weight_bytes)):
            raise ValueError('chain serving per-rank weight bytes must cover every rank')
        self._validate_model(pipeline, model)
        if (type(chunk_size) is not int
                or not 1 <= chunk_size <= 4096
                or chunk_size > pipeline.context):
            raise ValueError('chain serving chunk size must fit the session context')
        self.pipeline = pipeline
        self.alias = alias
        self.identity = bytes(identity)
        self.stage_layers = [list(pair) for pair in stage_layers]
        self.stage_quantizations = (None if stage_quantizations is None
                                    else [str(value) for value in stage_quantizations])
        self.stage_weight_bytes = (None if stage_weight_bytes is None
                                    else list(stage_weight_bytes))
        self.state = pipeline.state
        self.max_batch = pipeline.state.slots
        self.chunk_size = chunk_size
        self.models = {alias: model}
        self.identities = {alias: self.identity}
        self.aliases = [alias]
        if models is not None or identities is not None:
            if (not isinstance(models, dict) or not isinstance(identities, dict)
                    or alias not in models or alias not in identities
                    or len(models) != len(identities)
                    or len(models) != pipeline.state.models
                    or any(name == alias or not isinstance(name, str)
                           or not 1 <= len(name) <= 128 for name in models
                           if name != alias and name in models)
                    or set(models) != set(identities)):
                raise ValueError('chain serving model table must match the chain table')
            for name, extra in models.items():
                if name != alias:
                    self._validate_model(pipeline, extra)
            self.models = dict(models)
            self.identities = {name: bytes(value) for name, value in identities.items()}
            self.aliases = [alias] + [name for name in models if name != alias]
        self.model = model
        pipeline._completion_owner = object()  # type: ignore[attr-defined]

    def _validate_model(self, pipeline, model):
        for name in ('raw_tokens', 'chat_tokens'):
            if not callable(getattr(model, name, None)):
                raise ValueError(f'chain serving model needs {name}')
        eos = getattr(model, 'eos', ())
        if (not isinstance(eos, (list, tuple, set, frozenset)) or len(eos) > 256
                or any(type(token) is not int or not 0 <= token < pipeline.vocab_size
                       for token in eos)
                or type(getattr(model, 'max_context', None)) is not int
                or model.max_context > pipeline.context
                or type(getattr(model, 'quantization', None)) is not str
                or getattr(model, 'tokenizer', None) is None):
            raise ValueError('incompatible chain serving model surface')

    def switch_model(self, alias):
        """Swap the resident model; only the scheduler calls this while idle."""
        if alias not in self.models:
            raise ValueError('unknown chain model')
        index = self.aliases.index(alias)
        if self.pipeline.switch(index):
            self.model = self.models[alias]
            self.identity = self.identities[alias]
            self.alias = alias
            return True
        return False

    def prefill(self, tokens, slot):
        # Scalar: the scheduler wraps per-item prefill results itself.
        return self.pipeline.prefill(tokens, slot=slot)[0]

    def prefill_chunk(self, tokens, slot, start):
        # Scalar protocol sample; only the FINAL chunk's token is output.
        return self.pipeline.prefill_chunk(tokens, slot, start)[0]

    def step(self, tokens):
        # Single-slot protocol used by the scheduler when max_batch is 1.
        if not self.state.lengths[0]:
            return self.pipeline.prefill(tokens, slot=0)[0]
        if len(tokens) != 1:
            raise ValueError('single-slot chain decode requires one token')
        return self.pipeline.decode_batch(tokens, [0])[0]

    def decode(self, tokens, slots):
        return self.pipeline.decode_batch(tokens, slots)

    def release(self, slots=None):
        return self.pipeline.release([0] if slots is None else slots)

    def heartbeat(self):
        return self.pipeline.ping()

    def stop(self):
        return self.pipeline.close()

    def close(self):
        pipeline = self.pipeline
        if pipeline.failed:
            # A poisoned epoch cannot accept STOP; workers exit on their own.
            pipeline.closed = True
            return None
        return pipeline.close()


class ChainDistributedRuntime(DistributedRuntime):
    """HTTP serving over the three-rank chain; chain metadata stays truthful."""

    source_tier = 'gpu'
    kernel_backend = 'distributed-chain-broadcast'

    def __init__(self, pair, alias='model', queue_size=8, queue_timeout=120, *,
                 client_decode_turns=False, interleaved_prefill=False,
                 client_prefill_turns=False):
        if not isinstance(pair, ChainServingPair):
            raise ValueError('chain HTTP requires its explicit pair adapter')
        if (pair.pipeline.failed or pair.pipeline.closed):
            raise ValueError('chain HTTP requires a ready chain pipeline')
        super().__init__(pair, alias, queue_size, queue_timeout,
                         client_decode_turns=client_decode_turns,
                         interleaved_prefill=interleaved_prefill,
                         client_prefill_turns=client_prefill_turns)

    def model_manifest(self):
        raise ValueError('chain sessions cannot be exported as GPU model manifests')

    def manage(self, action, **options):
        if action != 'status' or options:
            raise ValueError('chain model management is not supported')
        pair = self.pair
        status = dict(device='distributed-cuda-chain',
                      context=pair.pipeline.context,
                      models=[dict(id=alias, residency='gpu' if alias == self.current_alias else 'host',
                                   quantization=pair.models[alias].quantization,
                                   model_sha256=pair.identities[alias].hex())
                            for alias in pair.aliases],
                      current_model=self.current_alias, model_switches=self.switches,
                      distributed=True, stage_layers=[list(bounds) for bounds in pair.stage_layers],
                      max_batch_size=self.max_batch, packed_prefill=False,
                      active=self.active, queued=self.requests.qsize(), completed=self.completed,
                      heartbeats=self.heartbeats, stream_buffer_events=STREAM_BUFFER_EVENTS,
                      client_admission_turns=True, client_decode_turns=self.client_decode_turns,
                      client_prefill_turns=self.client_prefill_turns,
                      interleaved_prefill=self.interleaved_prefill,
                      prefill_chunk_size=self.pair.chunk_size,
                      failed=self.failure is not None, closed=self.closed.is_set())
        if pair.stage_quantizations is not None:
            # Mixed-format chains (#601) report the per-rank precision and
            # per-rank weight bytes the planner actually priced.
            status['rank_quantizations'] = list(pair.stage_quantizations)
            if pair.stage_weight_bytes is not None:
                status['rank_weight_bytes'] = list(pair.stage_weight_bytes)
        return status

    def _execution_failed(self, primary):
        # A poisoned chain epoch is never returned to an owner for reuse.
        try:
            self.pair.close()
        except BaseException:
            primary.add_note('chain cleanup failed; retain pair and retry close')

    def close(self):
        super().close()
        self.pair.close()  # STOP closes the worker loops on every rank


def handler_for_chain(runtime, auth=None):
    """Reuse ordinary HTTP validation/auth while reporting chain health."""
    if not isinstance(runtime, ChainDistributedRuntime):
        raise ValueError('chain HTTP handler requires a chain runtime')
    from .server import handler_for
    base = handler_for(runtime, auth)

    class Handler(base):
        def do_GET(self):
            if self.path != '/health':
                return super().do_GET()
            if not self.authorize():
                return
            pipeline = runtime.pair.pipeline
            healthy = (not runtime.closed.is_set() and runtime.failure is None
                       and not pipeline.failed and not pipeline.closed)
            self.send_json(200 if healthy else 503,
                           dict(status='ok' if healthy else 'closed',
                                scope='distributed_chain',
                                stage_layers=[list(b) for b in runtime.pair.stage_layers],
                                identity=runtime.pair.identity.hex()))
    return Handler

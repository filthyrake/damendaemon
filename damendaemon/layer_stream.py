"""Synchronous, bounded host-weight streaming for experimental hybrid execution.

No pinned staging, asynchronous prefetch, or CPU layer computation. The caller
owns serialization and supplies the device synchronization/cache operations.
"""
from dataclasses import dataclass
import os


def enabled():
    value = os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD', '0')
    if value not in ('0', '1'):
        raise ValueError('DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD must be 0 or 1')
    return value == '1'


def validate(config, quantization, context, batch=1, use_graph=False, kernels=None):
    if not enabled():
        return
    if (os.environ.get('DAMENDAEMON_DEVICE', 'cuda') != 'cuda'
            or config.get('model_type') != 'qwen3_5_text'
            or quantization != 'none' or use_graph
            or (kernels or os.environ.get('DAMENDAEMON_KERNELS', 'auto')) != 'torch'
            or os.environ.get('DAMENDAEMON_HYBRID_PRECISION', 'float16') != 'float16'
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_VISION', '0') != '0'
            or os.environ.get('DAMENDAEMON_HYBRID_PREFILL', 'scan') != 'scan'
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT') is not None
            or os.environ.get('DAMENDAEMON_KV_CACHE_TOKENS', '0') != '0'):
        raise ValueError('hybrid offload requires unquantized CUDA hybrid text, eager torch, FP16 and --no-graph')
    if type(context) is not int or not 1 <= context <= min(2048, config['max_position_embeddings']):
        raise ValueError('hybrid offload context must fit model positions and be at most 2048')
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError('hybrid offload batch must be 1..4')


@dataclass(frozen=True)
class WeightPlan:
    fixed: tuple
    layers: tuple
    fixed_bytes: int
    layer_bytes: tuple

    @property
    def peak_bytes(self):
        return self.fixed_bytes + max(self.layer_bytes, default=0)


def plan(weights, layer_count):
    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError('invalid streamed layer count')
    fixed, layers = [], [[] for _ in range(layer_count)]
    for name, tensor in weights.items():
        if tensor.device.type not in ('cpu', 'meta') or not tensor.is_floating_point():
            raise ValueError('layer planning requires floating-point CPU or shape-only source weights')
        if name in ('embed', 'head', 'norm'):
            fixed.append(name)
        else:
            prefix, separator, _ = name.partition('.')
            if not separator or not prefix.isascii() or not prefix.isdecimal() or not 0 <= int(prefix) < layer_count:
                raise ValueError(f'unrecognized streamed weight: {name}')
            layers[int(prefix)].append(name)
    if set(fixed) != {'embed', 'head', 'norm'} or any(not names for names in layers):
        raise ValueError('streaming requires fixed weights and every layer')
    fixed_ids = {id(weights[name]) for name in fixed}
    def size(names, excluded=()):
        return sum(t.numel()*t.element_size() for identity, t in
                   {id(weights[name]): weights[name] for name in names}.items() if identity not in excluded)
    return WeightPlan(tuple(fixed), tuple(tuple(names) for names in layers), size(fixed),
                      tuple(size(names, fixed_ids) for names in layers))


class LayerWeightStore:
    def __init__(self, weights, layout, copy, synchronize, empty_cache):
        if any(t.device.type != 'cpu' for t in weights.values()):
            raise ValueError('layer execution requires actual CPU source weights')
        self.source, self.layout = weights, layout
        self.copy, self.synchronize, self.empty_cache = copy, synchronize, empty_cache
        self.resident = {}
        self.active = None
        self.failed = self.closed = False
        self.copied_bytes = self.transfers = self.peak_live_bytes = 0
        try:
            self._load(layout.fixed)
        except BaseException:
            self.failed = True
            self.resident.clear()
            self.empty_cache()
            raise

    def _load(self, names):
        aliases = {id(self.source[name]): tensor for name, tensor in self.resident.items()}
        added = []
        try:
            for name in names:
                identity = id(self.source[name])
                if identity not in aliases:
                    aliases[identity] = self.copy(self.source[name])
                    self.copied_bytes += self.source[name].numel()*self.source[name].element_size()
                    self.transfers += 1
                self.resident[name] = aliases[identity]
                added.append(name)
                self.peak_live_bytes = max(self.peak_live_bytes, self.resident_bytes)
        except BaseException:
            for name in added:
                self.resident.pop(name, None)
            aliases.clear()
            raise

    @property
    def resident_bytes(self):
        return sum(t.numel()*t.element_size() for t in {id(t):t for t in self.resident.values()}.values())

    def activate(self, layer):
        if self.closed or self.failed:
            raise RuntimeError('streamed weight store is closed or failed')
        if type(layer) is not int or not 0 <= layer < len(self.layout.layers):
            raise ValueError('invalid streamed layer')
        if layer == self.active:
            return
        try:
            if self.active is not None:
                self.synchronize()
                for name in self.layout.layers[self.active]:
                    self.resident.pop(name, None)
                self.active = None
                self.empty_cache()
            self._load(self.layout.layers[layer])
            self.active = layer
        except BaseException:
            self.failed = True
            self.empty_cache()
            raise

    def close(self):
        if self.closed:
            return
        # Keep ownership if synchronization fails; releasing potentially live
        # storage after a device failure is not a successful cleanup.
        self.synchronize()
        self.resident.clear()
        self.source = {}
        self.active = None
        self.closed = True
        self.empty_cache()

    def status(self):
        return dict(mode='synchronous-layer-streaming', active_layer=self.active,
                    resident_weight_bytes=self.resident_bytes, peak_live_weight_bytes=self.peak_live_bytes,
                    planned_peak_weight_bytes=self.layout.peak_bytes, copied_weight_bytes=self.copied_bytes,
                    tensor_transfers=self.transfers, failed=self.failed, closed=self.closed)

"""Layer-local hybrid ownership building block; no transport or serving admission.

Stages retain global layer indices. The first owns embedding, the last owns
final norm/head, and each allocates only its KV/recurrent/convolution layers.
A future coordinator must commit request cursors only after every stage succeeds.
"""
import copy

import torch

from .hybrid import HybridExecutor


def shard_model(model, start, end):
    c = model.config
    if (c.get('model_type') != 'qwen3_5_text' or model.quantization != 'none'
            or model.scales or getattr(model, 'vision_weights', None) is not None):
        raise ValueError('hybrid stage ownership currently requires unquantized text weights')
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= c['num_hidden_layers']):
        raise ValueError('invalid hybrid stage range')
    selected = {}
    for name, tensor in model.weights.items():
        prefix, separator, _ = name.partition('.')
        if separator and prefix.isascii() and prefix.isdecimal():
            if start <= int(prefix) < end:
                selected[name] = tensor
        elif name == 'embed' and start == 0:
            selected[name] = tensor
        elif name in ('norm', 'head') and end == c['num_hidden_layers']:
            selected[name] = tensor
        elif name not in ('embed', 'norm', 'head'):
            raise ValueError(f'unrecognized hybrid stage weight: {name}')
    if any(t.device.type != 'cpu' or not t.is_floating_point() for t in selected.values()):
        raise ValueError('hybrid stages require floating-point CPU source weights')
    for layer in range(start, end):
        if not any(name.startswith(f'{layer}.') for name in selected):
            raise ValueError('missing hybrid stage layer weights')
    required = ({'embed'} if start == 0 else set()) | ({'norm', 'head'} if end == c['num_hidden_layers'] else set())
    if not required <= selected.keys():
        raise ValueError('missing hybrid stage boundary weights')
    shard = copy.copy(model)
    shard.config = dict(c)  # Preserve global dimensions and layer numbering.
    shard.weights = selected
    shard.scales = {}
    shard.bytes = sum(t.numel()*t.element_size() for t in {id(t):t for t in selected.values()}.values())
    return shard


class HybridLayerStage(HybridExecutor):
    def __init__(self, model, start, end, max_batch=4):
        identity = getattr(model, '_hybrid_stage_identity', None)
        if identity is None:
            identity = model._hybrid_stage_identity = object()
        self.source_identity = identity
        self.closed = False
        self.first = start == 0
        self.last = end == model.config['num_hidden_layers']
        super().__init__(shard_model(model, start, end), use_graph=False,
                         max_batch=max_batch, kernels='torch', _layer_range=(start, end))

    @property
    def actual_device(self):
        if self.closed:
            raise RuntimeError('hybrid stage is closed')
        return next(iter(self.w.values())).device

    def synchronize(self):
        if self.closed:
            return
        device = self.actual_device
        if device.type != 'cpu':
            getattr(torch, device.type).synchronize(device)

    def close(self):
        if self.closed:
            return
        self.synchronize()
        device = self.actual_device
        self.w.clear()
        self.kv.clear()
        self.conv.clear()
        self.state.clear()
        self.model.weights.clear()
        for name in ('cos', 'sin', 'arange'):
            delattr(self, name)
        self.closed = True
        if device.type != 'cpu':
            api = getattr(torch, device.type)
            with api.device(device):
                api.empty_cache()

    def forward_layers(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError('hybrid stage is closed')
        return super().forward_layers(*args, **kwargs)

    def embedding(self, ids):
        if self.closed:
            raise RuntimeError('hybrid stage is closed')
        if not self.first:
            raise ValueError('only the first hybrid stage owns embedding')
        return super().embedding(ids)

    def output_logits(self, hidden):
        if self.closed:
            raise RuntimeError('hybrid stage is closed')
        if not self.last:
            raise ValueError('only the last hybrid stage owns the output head')
        return super().output_logits(hidden)

    def forward_embeddings(self, *args, **kwargs):
        raise ValueError('hybrid stages require coordinator-owned layer execution and cursor commit')

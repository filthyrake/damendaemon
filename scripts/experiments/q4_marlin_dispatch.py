"""Validation-only Q4_0 Marlin prefill/batched decode, with extra residency.

Each executor owns its prepared matrices and workspace. Normal admission does
not account for this experiment; a conservative free-memory gate is additional
protection, not a production memory contract.
"""
import hashlib
from pathlib import Path

import torch
from damendaemon import quantization
from .q4_marlin import Library


def install_validation_dispatch(path):
    path = str(Path(path).resolve())
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('Q4 Marlin model experiment requires the A2')
    original = quantization.Linear

    class Q4MarlinLinear(original):
        def __init__(self, weights, *args, **kwargs):
            super().__init__(weights, *args, **kwargs)
            self.experimental_q4_marlin = {}
            self.experimental_q4_marlin_bytes = 0
            self.experimental_q4_marlin_calls = 0
            # Reuse the existing narrow Mistral-style Q4/mixed/context scope.
            # The qualified single-row residual path remains selected.
            if self.q4_decode is None or self.formats or self.awq:
                return
            names = ['head'] + [f'{layer}.{part}' for layer in range(32)
                                for part in ('qkv.weight', 'out', 'gate_up', 'down')]
            if any(name not in weights or weights[name].dtype != torch.uint8 for name in names):
                raise ValueError('Q4 Marlin experiment requires the exact qualified projection set')
            selected = {id(weights[name]):weights[name] for name in names}
            needed = sum(w.numel()+self.scales[key].numel()*2+(w.shape[1]*2//32)*(w.shape[0]//8)*4
                         for key,w in selected.items())
            # Conversion creates integer unpacking/packing intermediates. Leave
            # additional room for KV, graph pools and ordinary runtime scratch.
            temporary = max(w.numel()*32 for w in selected.values())
            free, _ = torch.cuda.mem_get_info()
            if free < needed+temporary+2*1024**3:
                raise MemoryError('insufficient free memory for experimental Q4 Marlin preparation')
            self.experimental_q4_marlin_library = Library(path)
            library = self.experimental_q4_marlin_library
            for key, weight in selected.items():
                self.experimental_q4_marlin[key] = library.prepare(weight, self.scales[key])
            self.experimental_q4_marlin_bytes = sum(m.bytes for m in self.experimental_q4_marlin.values())
            if self.experimental_q4_marlin_bytes != needed:
                raise AssertionError('Q4 Marlin prepared byte estimate mismatch')

        def __call__(self, x, weight, bias=None):
            matrix = self.experimental_q4_marlin.get(id(weight))
            if (matrix is not None and bias is None and x.dtype == torch.float16
                    and x.numel()//x.shape[-1] > 1):
                shape = x.shape[:-1]
                self.experimental_q4_marlin_calls += 1
                result = self.experimental_q4_marlin_library.linear(x.reshape(-1, x.shape[-1]), matrix)
                return result.reshape(*shape, matrix.outputs)
            return super().__call__(x, weight, bias)

    quantization.Linear = Q4MarlinLinear
    return dict(kind='experimental Q4_0 Marlin rows greater than one', abi=1,
                library_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(Path(__file__).with_name('q4_marlin.py').read_bytes()).hexdigest(),
                production_admission_accounts_for_extra_weights=False)

"""Isolated all-row Marlin model experiment replacing canonical GPU Q4 matrices.

Host weights stay canonical. Normal admission does not account for this
experiment's layout overhead; the local free-memory gate is not a production
memory contract. No fallback may interpret a replaced matrix as canonical Q4.
"""
import hashlib
from pathlib import Path

import torch
from dllm import quantization
from dllm.qwen2 import Executor
from .q4_marlin_direct import DirectLibrary


def install_validation_dispatch(path):
    path = str(Path(path).resolve())
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('Q4 Marlin replacement experiment requires the A2')
    original = quantization.Linear
    original_executor_init = Executor.__init__

    class ReplacedQ4Linear(original):
        def __init__(self, weights, *args, **kwargs):
            super().__init__(weights, *args, **kwargs)
            self.experimental_q4_marlin = {}
            self.experimental_q4_marlin_bytes = 0
            self.experimental_q4_marlin_calls = 0
            self.experimental_q4_marlin_total_bytes = 0
            self.experimental_q4_marlin_replaced_bytes = 0
            if self.q4_decode is None or self.formats or self.awq:
                return
            names = ['head']+[f'{layer}.{part}' for layer in range(32)
                             for part in ('qkv.weight','out','gate_up','down')]
            if any(name not in weights or weights[name].dtype!=torch.uint8 for name in names):
                raise ValueError('Q4 Marlin replacement requires the qualified projection set')
            identities = {id(weights[name]) for name in names}
            if (len(identities)!=len(names) or id(weights['embed']) in identities
                    or len({id(self.scales[key]) for key in identities})!=len(names)):
                raise ValueError('Q4 Marlin replacement requires independent projection/scale buffers')
            overhead = sum((weights[name].shape[1]*2//32)*(weights[name].shape[0]//8)*4 for name in names)
            temporary = max(weights[name].numel()+2*self.scales[id(weights[name])].numel()*2 for name in names)
            free, _ = torch.cuda.mem_get_info()
            if free < overhead+temporary+512*1024**2:
                raise MemoryError('insufficient memory for experimental Q4 Marlin layout replacement')
            library = DirectLibrary(path)
            self.experimental_q4_marlin_library = library
            self.q4_decode = None
            for name in names:
                weight = weights[name]
                key = id(weight)
                scales = self.scales.pop(key)
                old_bytes = weight.numel()+scales.numel()*scales.element_size()
                prepared = library.prepare(weight,scales)
                weights[name] = prepared.weight
                self.experimental_q4_marlin[id(prepared.weight)] = prepared
                self.experimental_q4_marlin_total_bytes += prepared.bytes
                self.experimental_q4_marlin_replaced_bytes += old_bytes
                del weight, scales
            self.experimental_q4_marlin_bytes = self.experimental_q4_marlin_total_bytes-self.experimental_q4_marlin_replaced_bytes
            if self.experimental_q4_marlin_bytes!=overhead:
                raise AssertionError('Q4 Marlin layout overhead estimate mismatch')

        def __call__(self, x, weight, bias=None):
            matrix = self.experimental_q4_marlin.get(id(weight))
            if matrix is not None:
                if bias is not None:
                    raise ValueError('qualified Q4 Marlin replacement has no projection biases')
                self.experimental_q4_marlin_calls += 1
                shape = x.shape[:-1]
                result = self.experimental_q4_marlin_library.linear(x.reshape(-1,x.shape[-1]),matrix)
                return result.reshape(*shape,matrix.outputs)
            return super().__call__(x,weight,bias)

    def initialize(self,*args,**kwargs):
        original_executor_init(self,*args,**kwargs)
        if getattr(getattr(self,'quant_linear',None),'experimental_q4_marlin_replaced_bytes',0):
            self.q4_decode = None
            self.kernel_backend = self.kernel_backend.replace('+cuda-q4-residual','')+'+experimental-q4-marlin-replaced'
    quantization.Linear = ReplacedQ4Linear
    Executor.__init__ = initialize
    return dict(kind='experimental all-row Q4_0 Marlin layout replacement',abi=1,
                library_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ('q4_marlin_replace.py','q4_marlin_direct.py','q4_marlin_repack.py','q4_marlin.py')},
                replaces_canonical_gpu_weights=True,host_weights_unchanged=True,
                production_admission_accounts_for_extra_weights=False)

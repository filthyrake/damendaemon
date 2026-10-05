"""Explicit whole-model experiment; replace GPU packing before graph capture."""
import hashlib
from pathlib import Path
import torch
from dllm import quantization
from .awq_marlin import Library


def install_validation_dispatch(path):
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('Marlin model experiment is qualified only on the A2')
    library = Library(path)
    original = quantization.Linear

    class MarlinLinear(original):
        def __init__(self, weights, *args, **kwargs):
            super().__init__(weights, *args, **kwargs)
            if not self.awq:
                return
            # Executor now releases transfer aliases before this constructor.
            # Account for one replacement plus conversion temporaries.
            matrices = list(self.awq.values())
            replacement = max(sum(t.numel()*t.element_size()
                                  for t in (m.weight, m.scales, m.zeros)) for m in matrices)
            scratch_bytes = 4*library.sms*64*256*4 + 4*library.sms*4
            temporary = max(m.inputs*m.outputs*2 for m in matrices)
            free, _ = torch.cuda.mem_get_info()
            if free < replacement + scratch_bytes + temporary + 512*1024**2:
                raise MemoryError('insufficient memory for experimental Marlin repacking peak')
            # Charge shared persistent scratch to the existing AWQ dense
            # projection reserve; general activation reserve stays separate.
            if scratch_bytes > temporary:
                raise MemoryError('Marlin scratch exceeds admitted packed workspace')
            device = matrices[0].weight.device
            self.native_workspace = (
                torch.empty(4*library.sms*64*256, dtype=torch.float32, device=device),
                torch.zeros(4*library.sms, dtype=torch.int32, device=device))
            del matrices
            prepared = {}
            for name in list(weights):
                key = id(weights[name])
                if key not in self.awq:
                    continue
                matrix = self.awq.pop(key)
                bias_name = name.removesuffix('.weight') + '.bias'
                value = library.prepare(matrix, weights.get(bias_name), self.native_workspace)
                weights[name] = value.weight
                self.scales.pop(key)
                self.scales[id(value.weight)] = value.scales
                if value.bias is not None:
                    weights[bias_name] = value.bias
                prepared[id(value.weight)] = value
                del matrix
            self.awq = prepared
            self.native_workspace_bytes = scratch_bytes
            self.native_repack_peak_reserve = replacement + scratch_bytes + temporary

        def __call__(self, x, weight, bias=None):
            matrix = self.awq.get(id(weight))
            if matrix is not None:
                if bias is not matrix.bias:
                    raise ValueError('native packed bias identity mismatch')
                return library.linear(x, matrix)
            return super().__call__(x, weight, bias)

    quantization.Linear = MarlinLinear
    return dict(kind='experimental independent Marlin C ABI', abi=2,
                library_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                shared_workspace_bytes=4*library.sms*64*256*4+4*library.sms*4,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

"""Bounded control: FP16 prompt projections, FP32 decoding/state/nonlinear math."""
import runpy
from scripts.experiments import replay_hybrid_fp32
from damendaemon import hybrid

original_forward = hybrid.HybridExecutor.forward_embeddings
original_init = hybrid.HybridExecutor.__init__


def forward(self, x, positions, slots, prefill, prefix_length=0, return_logits=True):
    self._diagnostic_prefill = prefill
    return original_forward(self, x, positions, slots, prefill, prefix_length, return_logits)


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    original_linear = self.linear

    def linear(x, weight, bias=None):
        if self._diagnostic_prefill and weight is not self.w['head']:
            return original_linear(x.half(), weight, bias).float()
        return original_linear(x, weight, bias)

    self.linear = linear
    self.kernel_backend += '+diagnostic-fp16-prefill-projections'


hybrid.HybridExecutor.forward_embeddings = forward
hybrid.HybridExecutor.__init__ = initialize
runpy.run_path('scripts/replay_reference.py', run_name='__main__')

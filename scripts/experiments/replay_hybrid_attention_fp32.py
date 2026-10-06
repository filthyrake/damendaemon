"""Bounded precision ablation: FP32 full attention, ordinary hybrid/MLP math.

Only a <=512-context diagnostic. Does not change production admission/defaults.
"""
import inspect
import runpy
import textwrap
import torch
from dllm import hybrid

torch.backends.cuda.matmul.allow_tf32 = False


def replace_method(name, old, new, additional=()):
    source = textwrap.dedent(inspect.getsource(getattr(hybrid.HybridExecutor, name)))
    for before, after in [(old, new), *additional]:
        if source.count(before) != 1:
            raise ValueError(f'observation source changed: {name}')
        source = source.replace(before, after)
    namespace = vars(hybrid).copy()
    exec(compile(source, '<attention precision ablation>', 'exec'), namespace)
    setattr(hybrid.HybridExecutor, name, namespace[name])


replace_method('forward_embeddings', "h = self.norm(x, self.w[f'{i}.norm1'])",
               "h = self.norm(x, self.w[f'{i}.norm1'], torch.float32 if kind == 'full_attention' else torch.float16)")
replace_method('rotate', 'return torch.cat((rotated, rest), -1).to(self.compute_dtype)',
               'return torch.cat((rotated, rest), -1)')
replace_method('attention', "self.linear(out.reshape(n, length, -1).to(self.compute_dtype), w[f'{i}.o'])",
               "self.linear(out.reshape(n, length, -1), w[f'{i}.o'])")
replace_method('__init__',
               'self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=self.compute_dtype, device=self.device)',
               'self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=torch.float32, device=self.device)',
               additional=[('full * 2*self.kh*self.context*self.dim*cache_element_bytes', 'full * 2*self.kh*self.context*self.dim*4')])
original_init = hybrid.HybridExecutor.__init__


def initialize(self, model, *args, **kwargs):
    if model.max_context > 512:
        raise ValueError('precision ablation limited to context <=512')
    original_init(self, model, *args, **kwargs)
    self.kernel_backend += '+diagnostic-attention-fp32'


hybrid.HybridExecutor.__init__ = initialize
runpy.run_path('scripts/replay_reference.py', run_name='__main__')

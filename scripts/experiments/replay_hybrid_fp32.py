"""Small-context FP32 arithmetic control with packed Q8 storage; not a default."""
import inspect
import runpy
import textwrap
import torch
from dllm import hybrid

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def replace(name, replacements):
    source = textwrap.dedent(inspect.getsource(getattr(hybrid.HybridExecutor, name)))
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f'diagnostic source changed: {name}: {before}')
        source = source.replace(before, after)
    namespace = vars(hybrid).copy()
    exec(compile(source, '<hybrid FP32 diagnostic>', 'exec'), namespace)
    setattr(hybrid.HybridExecutor, name, namespace[name])


replace('norm', [('dtype=torch.float16', 'dtype=torch.float32')])
replace('rotate', [('return torch.cat((rotated, rest), -1).half()',
                    'return torch.cat((rotated, rest), -1)')])
replace('linear_attention', [
    ('self.eps, torch.float16, gate=z', 'self.eps, torch.float32, gate=z'),
    ("out.reshape(n, length, -1).half(), w[f'{i}.out']", "out.reshape(n, length, -1), w[f'{i}.out']")])
replace('attention', [("out.reshape(n, length, -1).half(), w[f'{i}.o']", "out.reshape(n, length, -1), w[f'{i}.o']")])
replace('forward_embeddings', [("(F.silu(gate.float())*up.float()).half()", "F.silu(gate.float())*up.float()")])
replace('__init__', [
    ('self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=torch.float16, device=self.device)',
     'self.kv[i] = torch.zeros((2, max_batch, self.kh, self.context, self.dim), dtype=torch.float32, device=self.device)'),
    ("self.conv[i] = torch.zeros((max_batch, self.conv_width, c['linear_conv_kernel_dim']), dtype=torch.float16, device=self.device)",
     "self.conv[i] = torch.zeros((max_batch, self.conv_width, c['linear_conv_kernel_dim']), dtype=torch.float32, device=self.device)"),
    ('full * 2*self.kh*self.context*self.dim*2', 'full * 2*self.kh*self.context*self.dim*4'),
    ("self.conv_width*c['linear_conv_kernel_dim']*2", "self.conv_width*c['linear_conv_kernel_dim']*4")])
original_init = hybrid.HybridExecutor.__init__


def initialize(self, model, *args, **kwargs):
    if model.max_context > 512 or model.quantization != 'q8_0':
        raise ValueError('FP32 arithmetic control requires Q8 and context <=512')
    original_init(self, model, *args, **kwargs)
    self.kernel_backend += '+diagnostic-fp32'


hybrid.HybridExecutor.__init__ = initialize
if __name__ == '__main__':
    runpy.run_path('scripts/replay_reference.py', run_name='__main__')

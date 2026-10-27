#!/usr/bin/env python3
"""#642 patch 3: shrink FlexAttention tiles for 100 KB-smem consumer GPUs.

Upstream default flex tiles at head_dim 256 compile a backward kernel
(triton_tem_fused_flex_attention_backward_2) needing 114,688 B of shared
memory — over the RTX 3090's 101,376 B hardware limit (A100-sized tiles).
FlexAttention accepts per-call kernel_options; this patch shadows the
imported flex_attention in tokengate.e2e_gate.attn.flex_attention_utils
with a wrapper that defaults BLOCK_M/BLOCK_N to 64 and num_stages to 2
(same math, smaller tiles). Honors TOKENGATE_FLEX_KERNEL_OPTIONS to
override from the environment (JSON dict).
"""
import json
import os
import pathlib

path = pathlib.Path('/home/damen/tokengate-640/tokengate/e2e_gate/attn/flex_attention_utils.py')
source = path.read_text()
old = "from torch.nn.attention.flex_attention import flex_attention, create_block_mask"
new = old + """

# [642 patch] head_dim-256 flex tiles exceed consumer-GPU shared memory
# (3090: 101,376 B limit vs ~112 KB required by the default backward). Shadow
# the imported flex_attention with a tile-shrinking wrapper; same math, block
# sizes only. Override with TOKENGATE_FLEX_KERNEL_OPTIONS (JSON dict).
_flex_attention_upstream = flex_attention
try:
    _flex_kernel_options = json.loads(
        os.environ.get('TOKENGATE_FLEX_KERNEL_OPTIONS',
                        '{"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2}'))
except json.JSONDecodeError as exc:
    raise RuntimeError('TOKENGATE_FLEX_KERNEL_OPTIONS must be a JSON dict') from exc


def flex_attention(*args, **kwargs):
    kwargs.setdefault('kernel_options', _flex_kernel_options)
    return _flex_attention_upstream(*args, **kwargs)"""
assert source.count(old) == 1, f'import site not unique: {source.count(old)}'
path.write_text(source.replace(old, new))
print('patched', path)

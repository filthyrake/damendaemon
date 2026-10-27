#!/usr/bin/env python3
"""#642 patch 4: correct inductor flex-backward tile keys.

torch's own OutOfMemoryError message (torch/_inductor/kernel/flex/common.py)
documents that the COMPILED flex path reads kernel_options keys
bwd_BLOCK_M1/N1/M2/N2, bwd_num_stages, bwd_num_warps (and fwd_*); the plain
BLOCK_M/BLOCK_N keys only configure the eager kernel — which is why patch 3
left the backward requirement unchanged (114,688 B vs 101,376 B on the 3090).
This patch sets the backward tiles to torch's documented low-smem example
(32/32/32/32, stages 1) alongside the eager keys.
"""
import pathlib

path = pathlib.Path('/home/damen/tokengate-640/tokengate/e2e_gate/attn/flex_attention_utils.py')
source = path.read_text()
old = """    _flex_kernel_options = _json.loads(
        _os.environ.get('TOKENGATE_FLEX_KERNEL_OPTIONS',
                        '{"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2}'))"""
new = """    _flex_kernel_options = _json.loads(
        _os.environ.get('TOKENGATE_FLEX_KERNEL_OPTIONS',
                        '{"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2, '
                        '"bwd_BLOCK_M1": 32, "bwd_BLOCK_N1": 32, '
                        '"bwd_BLOCK_M2": 32, "bwd_BLOCK_N2": 32, '
                        '"bwd_num_stages": 1, "bwd_num_warps": 4}'))"""
assert source.count(old) == 1, f'not unique: {source.count(old)}'
path.write_text(source.replace(old, new))
print('patched', path)

#!/usr/bin/env python3
"""#642 patch 3 fixup: self-contained imports in the flex tile wrapper."""
import pathlib

path = pathlib.Path('/home/damen/tokengate-640/tokengate/e2e_gate/attn/flex_attention_utils.py')
source = path.read_text()
old = '''_flex_attention_upstream = flex_attention
try:
    _flex_kernel_options = json.loads(
        os.environ.get('TOKENGATE_FLEX_KERNEL_OPTIONS',
                        '{"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2}'))
except json.JSONDecodeError as exc:
    raise RuntimeError('TOKENGATE_FLEX_KERNEL_OPTIONS must be a JSON dict') from exc'''
new = '''_flex_attention_upstream = flex_attention
import json as _json, os as _os
try:
    _flex_kernel_options = _json.loads(
        _os.environ.get('TOKENGATE_FLEX_KERNEL_OPTIONS',
                        '{"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2}'))
except _json.JSONDecodeError as exc:
    raise RuntimeError('TOKENGATE_FLEX_KERNEL_OPTIONS must be a JSON dict') from exc'''
assert source.count(old) == 1, f'not unique: {source.count(old)}'
path.write_text(source.replace(old, new))
print('fixed imports')

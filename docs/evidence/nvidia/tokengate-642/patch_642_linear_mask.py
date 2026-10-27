#!/usr/bin/env python3
"""#642 patch 2: linear-attention layers must not receive the 4D kv mask.

Upstream bug (the #640 'stale 4D mask' blocker, train-path manifestation):
gate_decode's decode loop passes the additive kv-width mask `dm`/`am`
[B, 1, 1, cache_next+1] to EVERY layer above the gate. Full-attention layers
consume it correctly, but Qwen3_5 linear-attention (DeltaNet) layers interpret
`attention_mask` as a padding mask over recurrent/sequence state —
apply_mask_to_padding_states raises 'size of tensor a (1024) must match b
(2050)' (modeling_qwen3_5.py:234). Linear layers keep no kv cache, so the
correct mask for them is None.
"""
import pathlib

path = pathlib.Path('/home/damen/tokengate-640/tokengate/e2e_gate/decode/gate_decode.py')
source = path.read_text()
old = """                am = fds.mask_for(li, cache_next + 1) if fds is not None else dm
                hs = inner.layers[li](
                    hs, attention_mask=am.to(hd),"""
new = """                am = fds.mask_for(li, cache_next + 1) if fds is not None else dm
                # [642 patch] linear-attention (DeltaNet) layers hold no kv
                # cache: the 4D kv-width mask breaks their padding-state
                # masking (apply_mask_to_padding_states). They take None.
                _lin = getattr(inner.layers[li], 'block_type',
                               getattr(inner.layers[li], 'layer_type',
                                       'full_attention')) == 'linear_attention'
                hs = inner.layers[li](
                    hs, attention_mask=None if _lin else am.to(hd),"""
assert source.count(old) == 1, f'decode call site not unique: {source.count(old)}'
path.write_text(source.replace(old, new))
print('patched', path)

#!/usr/bin/env python3
"""#642 patch: route hybrid l1d eval through prune_below_gate.

Upstream bug (train-path manifestation of the #640 hybrid blocker): with the
gate on a linear-attention layer (l1d lane), layers 0..gate_layer hold no KV
cache, so gate_decode.generate_batch's default prune_below_gate=False path
raises StopIteration at gate_decode.py:147 (`next(cl for cl in
cache.layers[:gate_layer+1] ...)`) during in-training gated eval. The
prune_below_gate=True prefill exists precisely for this case (#640 used it in
tg-hybrid-niah.py). Patch: pass prune_below_gate=True exactly when no
full-attention layer exists at or below the gate layer.
"""
import pathlib

path = pathlib.Path('/home/damen/tokengate-640/tokengate/e2e_gate/train/train_e2e_gate.py')
source = path.read_text()
old = """                        gate_layer, K_eval, max_gen_tokens,
                        decode_phase2_form=True,
                    )"""
new = """                        gate_layer, K_eval, max_gen_tokens,
                        decode_phase2_form=True,
                        # [642 patch] hybrid lanes whose gate layer sits on a
                        # linear-attention (DeltaNet) layer have no KV cache at
                        # or below the gate: the default prune_below_gate=False
                        # route StopIteration's at gate_decode.py:147. Route
                        # those lanes through the prune-below-gate prefill.
                        prune_below_gate=not any(
                            getattr(_l, 'block_type',
                                    getattr(_l, 'layer_type', 'full_attention')) == 'full_attention'
                            for _l in model.model.layers[:gate_layer + 1]),
                    )"""
assert source.count(old) == 1, f'call site not unique: {source.count(old)}'
path.write_text(source.replace(old, new))
print('patched', path)

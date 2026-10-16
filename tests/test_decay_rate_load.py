"""Corrupt decay-rate checkpoints must fail at load, not silently zero recurrence.

`A = -A_log.exp()`: an `A_log` above exp's overflow threshold yields an infinite
decay rate, and `exp(step*A)` then zeroes the recurrent state every token —
silent garbage output rather than an error. Also covers the hybrid linear
attention q/k L2 norm, whose overflow previously normalized to zeros.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from dllm.hybrid import _l2_normalize


def rewrite_checkpoint_a_log(directory, value):
    """Set every A_log entry in a saved checkpoint's shards to `value`."""
    for shard in sorted(Path(directory).glob('*.safetensors')):
        tensors = load_file(str(shard))
        touched = False
        for name in list(tensors):
            if 'A_log' in name:
                tensors[name] = torch.full_like(tensors[name], value)
                touched = True
        if touched:
            save_file(tensors, str(shard))
        return touched
    raise AssertionError('no safetensors shard found')


class MambaDecayLoadTests(unittest.TestCase):
    def test_mamba_rejects_overflowing_a_log_before_weights(self):
        from transformers import MambaConfig, MambaForCausalLM
        from tokenizers import Tokenizer, models
        from dllm.mamba import MambaWeights
        with TemporaryDirectory() as directory:
            reference = MambaForCausalLM(MambaConfig(hidden_size=8, state_size=4,
                num_hidden_layers=1, expand=2, conv_kernel=4, time_step_rank=2,
                vocab_size=16, eos_token_id=1))
            reference.save_pretrained(directory)
            Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]')).save(
                str(Path(directory) / 'tokenizer.json'))
            self.assertTrue(rewrite_checkpoint_a_log(directory, 100.0))
            with self.assertRaisesRegex(ValueError, 'nonfinite Mamba decay rates'):
                MambaWeights(directory, 16)

    def test_mamba2_rejects_overflowing_a_log_before_weights(self):
        from transformers import Mamba2Config, Mamba2ForCausalLM
        from tokenizers import Tokenizer, models
        from dllm.mamba2 import Mamba2Weights
        with TemporaryDirectory() as directory:
            reference = Mamba2ForCausalLM(Mamba2Config(hidden_size=8, state_size=4,
                num_hidden_layers=1, expand=2, conv_kernel=4, num_heads=4,
                head_dim=4, n_groups=1, vocab_size=16, eos_token_id=1))
            reference.save_pretrained(directory)
            Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]')).save(
                str(Path(directory) / 'tokenizer.json'))
            self.assertTrue(rewrite_checkpoint_a_log(directory, 100.0))
            with self.assertRaisesRegex(ValueError, 'nonfinite Mamba-2 decay rates'):
                Mamba2Weights(directory, 16)


class HybridL2NormTests(unittest.TestCase):
    def test_overflow_rejected_and_ordinary_matches_previous_formula(self):
        value = torch.zeros(1, 1, 3, 4); value[..., 0] = 1e20
        with self.assertRaisesRegex(RuntimeError, 'nonfinite hybrid linear-attention q/k norm'):
            _l2_normalize(value)
        ordinary = torch.randn(2, 3, 4)
        expected = ordinary.float() * torch.rsqrt(
            ordinary.float().square().sum(-1, keepdim=True) + 1.e-6)
        torch.testing.assert_close(_l2_normalize(ordinary), expected)


if __name__ == '__main__':
    unittest.main()

"""CPU checks for normalized static rotary schedules."""
import unittest
import torch
from dllm.config import decoder_config
from dllm.rotary import inverse_frequencies


class RotaryTests(unittest.TestCase):
    def test_adapter_linear_aliases_reach_frequency_construction(self):
        for family in ('starcoder2', 'olmoe'):
            for key in ('type', 'rope_type'):
                source = dict(model_type=family, hidden_size=8, intermediate_size=16,
                    num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
                    max_position_embeddings=128, vocab_size=32, rms_norm_eps=1e-5,
                    hidden_act='gelu_pytorch_tanh' if family == 'starcoder2' else 'silu',
                    num_experts=4, num_experts_per_tok=2, rope_theta=10000.,
                    rope_scaling={key:'linear', 'factor':4})
                with self.subTest(family=family, key=key):
                    parsed = decoder_config(source)
                    expected = torch.tensor([.25, .0025])
                    torch.testing.assert_close(inverse_frequencies(parsed, 'cpu'), expected)

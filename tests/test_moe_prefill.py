import unittest
from dllm.moe_prefill import qualified_scope, use_grouped_prefill


class MoEPrefillScopeTests(unittest.TestCase):
    config = dict(model_type='olmoe', hidden_size=2048, intermediate_size=1024,
                  num_hidden_layers=16, num_attention_heads=16,
                  num_key_value_heads=16, num_experts=64, num_experts_per_tok=8,
                  vocab_size=50304, norm_topk_prob=False)

    def test_only_measured_geometry_and_device(self):
        args = ('NVIDIA A2', 'q8_0', 2048, 4)
        self.assertTrue(qualified_scope(self.config, *args))
        for key in self.config:
            self.assertFalse(qualified_scope({**self.config, key: None}, *args))
        self.assertFalse(qualified_scope({**self.config, 'clip_qkv': .75}, *args))
        for device, quant, context, batch in [
                ('NVIDIA A30', 'q8_0', 2048, 4), ('NVIDIA A2', 'none', 2048, 4),
                ('NVIDIA A2', 'q8_0', 2049, 4), ('NVIDIA A2', 'q8_0', 0, 4),
                ('NVIDIA A2', 'q8_0', 2048, 5), ('NVIDIA A2', 'q8_0', 2048, 0)]:
            self.assertFalse(qualified_scope(self.config, device, quant, context, batch))

    def test_longer_prompts_preserve_baseline(self):
        for tokens in (1, 4, 16, 32, 64, 128):
            self.assertTrue(use_grouped_prefill(True, tokens))
            self.assertFalse(use_grouped_prefill(False, tokens))
        for tokens in (0, 129, 256, 512, 1024, 2048):
            self.assertFalse(use_grouped_prefill(True, tokens))

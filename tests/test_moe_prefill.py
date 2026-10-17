import unittest
from damendaemon.moe_prefill import qualified_scope, use_grouped_prefill, use_packed_prefill


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

    def test_packed_batches_remain_within_measured_token_budget(self):
        for lengths in ([1,1], [1]*4, [31,65], [32]*4, [64,64], [31,31,65], [1,127]):
            self.assertTrue(use_packed_prefill(True,lengths))
            self.assertFalse(use_packed_prefill(False,lengths))
        for lengths in ([], [32], [1]*5, [65,65], [64,65], [0,32], [-1,32]):
            self.assertFalse(use_packed_prefill(True,lengths))

    def test_invalid_chunk_state_fails_before_touching_device_or_lengths(self):
        from damendaemon.moe import OlmoeExecutor
        executor = object.__new__(OlmoeExecutor)
        executor.grouped_prefill = True
        executor.max_batch = 4
        executor.context = 128
        executor.lengths = [7, 0, 12, 0]
        before = executor.lengths.copy()
        for tokens,slot,start in [([1],0,-1), ([1],0,1.0), ([1],4,0),
                                  ([1],0,6), ([],0,0), ([1]*129,0,0), ([1]*122,0,7)]:
            with self.assertRaises(ValueError):
                executor.prefill_chunk(tokens,slot,start)
            self.assertEqual(executor.lengths,before)
        executor.grouped_prefill = False
        with self.assertRaisesRegex(ValueError,'not supported'):
            executor.prefill_chunk([1],0,0)
        self.assertEqual(executor.lengths,before)

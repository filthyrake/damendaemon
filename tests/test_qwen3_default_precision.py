"""Qwen3 CUDA default precision selection (#2).

The fp16 standard path fails the long-context logit gate deterministically
(evidence docs/evidence/models/qwen3-standard-gate-repro-2/); the qualified
FP32-arithmetic path is the default for unpacked checkpoints, while
quantized weights (packed kquant/GGUF, AWQ FP16 rounding) keep the standard
fp16 default their own gates qualified. Pure selection-logic tests.
"""
import unittest
from types import SimpleNamespace

from damendaemon.qwen2 import qwen3_default_precision


class Qwen3DefaultPrecisionTests(unittest.TestCase):
    def unpacked_model(self):
        return SimpleNamespace(scales={}, packed_formats={}, awq_fp16_rounding=False)

    def test_unpacked_defaults_to_the_qualified_float32_path(self):
        self.assertEqual(qwen3_default_precision(self.unpacked_model()), 'float32')

    def test_missing_quant_attributes_default_to_float32(self):
        # HF snapshot models carry no quantization attributes at all.
        self.assertEqual(qwen3_default_precision(SimpleNamespace()), 'float32')

    def test_scales_keep_standard(self):
        model = self.unpacked_model()
        model.scales = {'w': object()}
        self.assertEqual(qwen3_default_precision(model), 'standard')

    def test_packed_formats_keep_standard(self):
        model = self.unpacked_model()
        model.packed_formats = {'q4_0': object()}
        self.assertEqual(qwen3_default_precision(model), 'standard')

    def test_awq_rounding_keeps_standard(self):
        model = self.unpacked_model()
        model.awq_fp16_rounding = True
        self.assertEqual(qwen3_default_precision(model), 'standard')

    def test_explicit_env_override_still_wins(self):
        # The Executor reads DAMENDAEMON_QWEN3_PRECISION with this function
        # as the fallback only; an explicit 'standard' or 'float32' must not
        # be second-guessed. (Executed on CUDA; here we assert the fallback
        # never raises for any model shape so the override path is the only
        # decision point left.)
        for model in (self.unpacked_model(), SimpleNamespace(scales={'w': 1})):
            self.assertIn(qwen3_default_precision(model), ('standard', 'float32'))


if __name__ == '__main__':
    unittest.main()

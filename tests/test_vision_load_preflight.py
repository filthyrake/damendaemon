"""Unsupported vision context must not materialize either host checkpoint."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dllm.models import load


class VisionLoadPreflightTests(unittest.TestCase):
    def test_wrapper_keeps_geometry_and_context_contract(self):
        from dllm.vision_model import VisionModelWeights
        config = {'model_type': 'qwen3_5_text', 'hidden_size': 1024,
                  'num_hidden_layers': 24}
        for changed, context in (({'model_type': 'qwen3'}, 1024),
                                 ({'hidden_size': 2048}, 1024),
                                 ({'num_hidden_layers': 32}, 1024), ({}, 2048)):
            with self.subTest(changed=changed, context=context), self.assertRaises(ValueError):
                VisionModelWeights(SimpleNamespace(config={**config, **changed},
                    max_context=context, bytes=10), SimpleNamespace(bytes=20))

    def test_context_preflight_and_text_only_control(self):
        config = {'model_type': 'qwen3_5_text', 'hidden_size': 1024,
                  'num_hidden_layers': 24}
        for enabled, context in (('1', 2048), ('1', 1024), ('0', 2048)):
            with self.subTest(enabled=enabled, context=context), \
                    patch.dict(os.environ, DLLM_EXPERIMENTAL_VISION=enabled), \
                    patch('dllm.models.validate_device_model'), \
                    patch('dllm.models.is_gguf', return_value=False), \
                    patch('dllm.models.config', return_value=config), \
                    patch('dllm.qwen2.ModelWeights') as text, \
                    patch('dllm.vision_qwen35.VisionWeights') as vision:
                text.return_value = SimpleNamespace(config=config, max_context=context, bytes=10)
                vision.return_value = SimpleNamespace(bytes=20)
                if enabled == '1' and context > 1024:
                    with self.assertRaises(ValueError):
                        load('/unused', context)
                    text.assert_not_called()
                    vision.assert_not_called()
                else:
                    result = load('/unused', context)
                    text.assert_called_once_with('/unused', context, quantization=None)
                    if enabled == '1':
                        vision.assert_called_once_with('/unused')
                        self.assertEqual(result.bytes, 30)
                    else:
                        vision.assert_not_called()
                        self.assertIs(result, text.return_value)

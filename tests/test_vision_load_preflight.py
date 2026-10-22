"""Unsupported vision context must not materialize either host checkpoint."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from damendaemon.models import load


class VisionLoadPreflightTests(unittest.TestCase):
    def test_wrapper_keeps_geometry_and_context_contract(self):
        from damendaemon.vision_model import VisionModelWeights
        config = {'model_type': 'qwen3_5_text', 'hidden_size': 1024,
                  'num_hidden_layers': 24}
        for changed, context in (({'model_type': 'qwen3'}, 1024),
                                 ({'hidden_size': 1536}, 1024),
                                 ({'num_hidden_layers': 32}, 1024), ({}, 2048)):
            with self.subTest(changed=changed, context=context), self.assertRaises(ValueError):
                VisionModelWeights(SimpleNamespace(config={**config, **changed},
                    max_context=context, bytes=10), SimpleNamespace(bytes=20))

    def test_context_preflight_and_text_only_control(self):
        config = {'model_type': 'qwen3_5_text', 'hidden_size': 1024,
                  'num_hidden_layers': 24}
        for enabled, context in (('1', 2048), ('1', 1024), ('0', 2048)):
            with self.subTest(enabled=enabled, context=context), \
                    patch.dict(os.environ, DAMENDAEMON_EXPERIMENTAL_VISION=enabled), \
                    patch('damendaemon.models.validate_device_model'), \
                    patch('damendaemon.models.is_gguf', return_value=False), \
                    patch('damendaemon.models.config', return_value=config), \
                    patch('damendaemon.qwen2.ModelWeights') as text, \
                    patch('damendaemon.vision_qwen35.VisionWeights') as vision, \
                    patch('damendaemon.vision_qwen35.read_vision_config') as metadata:
                text.return_value = SimpleNamespace(config=config, max_context=context, bytes=10)
                vision.return_value = SimpleNamespace(bytes=20)
                if enabled == '1' and context > 1024:
                    with self.assertRaises(ValueError):
                        load('/unused', context)
                    text.assert_not_called()
                    vision.assert_not_called()
                    metadata.assert_not_called()
                else:
                    result = load('/unused', context)
                    text.assert_called_once_with('/unused', context, quantization=None)
                    if enabled == '1':
                        metadata.assert_called_once_with('/unused')
                        vision.assert_called_once_with('/unused')
                        self.assertEqual(result.bytes, 30)
                    else:
                        vision.assert_not_called()
                        metadata.assert_not_called()
                        self.assertIs(result, text.return_value)


class VisionMetadataTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.geometry = dict(depth=12,hidden_size=768,intermediate_size=3072,num_heads=12,
            out_hidden_size=1024,patch_size=16,spatial_merge_size=2,temporal_patch_size=2,
            in_channels=3,num_position_embeddings=2304,hidden_act='gelu_pytorch_tanh')

    def write(self, vision):
        import json
        (self.path/'config.json').write_text(json.dumps(dict(model_type='qwen3_5',vision_config=vision)))

    def test_bad_containers_and_numeric_types_reject_before_tensor_scan(self):
        from damendaemon.vision_qwen35 import VisionWeights
        cases = [None, [], 'vision', 12]
        for key, value in self.geometry.items():
            if type(value) is int:
                cases.extend({**self.geometry, key:bad} for bad in (float(value), str(value), True, None))
        cases.extend({**self.geometry, 'rope_parameters':bad}
                     for bad in ([], 'axial', 10000, {'rope_type':'axial','rope_theta':'10000'},
                                 {'rope_type':'axial','rope_theta':True}))
        for vision in cases:
            with self.subTest(vision=vision), patch('damendaemon.vision_qwen35.weight_files') as files:
                self.write(vision)
                with self.assertRaisesRegex(ValueError, 'vision'):
                    VisionWeights(self.path)
                files.assert_not_called()

    def test_valid_geometry_and_rotary_controls(self):
        from damendaemon.vision_qwen35 import read_vision_config, VisionWeights
        for rope in (None, {'rope_type':'axial','rope_theta':10000},
                     {'rope_type':'axial','rope_theta':10000.0}):
            vision = {**self.geometry, 'rope_parameters':rope}
            self.write(vision)
            self.assertEqual(read_vision_config(self.path), vision)
            # Valid metadata reaches the tensor boundary; no large weights needed.
            with patch('damendaemon.vision_qwen35.weight_files', side_effect=RuntimeError('tensor boundary')):
                with self.assertRaisesRegex(RuntimeError, 'tensor boundary'):
                    VisionWeights(self.path)

    def test_dispatch_rejects_before_text_and_vision_weights(self):
        for vision in ([], {**self.geometry, 'depth':12.0}, {**self.geometry, 'rope_parameters':[]}):
            self.write(vision)
            with patch.dict(os.environ, DAMENDAEMON_EXPERIMENTAL_VISION='1'), \
                    patch('damendaemon.models.validate_device_model'), \
                    patch('damendaemon.models.config', return_value=dict(model_type='qwen3_5_text',
                          hidden_size=1024,num_hidden_layers=24)), \
                    patch('damendaemon.qwen2.ModelWeights') as text, \
                    patch('damendaemon.vision_qwen35.VisionWeights') as images:
                with self.assertRaises(ValueError):
                    load(self.path, 1024)
                text.assert_not_called()
                images.assert_not_called()

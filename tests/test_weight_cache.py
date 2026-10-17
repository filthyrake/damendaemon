import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from damendaemon.weight_cache import PreparedCache, tensor_schema


class WeightCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(model_type='qwen2', hidden_size=32, num_attention_heads=2,
                           num_key_value_heads=1, intermediate_size=64,
                           vocab_size=64, num_hidden_layers=1)
        (self.root / 'config.json').write_text(json.dumps(self.config))
        self.source = self.root / 'model.safetensors'
        self.source.write_bytes(b'checkpoint identity fixture')

    def cache(self, mode='q4_0'):
        return PreparedCache(self.root / 'cache', self.root, self.config, mode, [self.source])

    def values(self, cache):
        tensors = {k: torch.ones(shape, dtype=dtype) for k, (shape, dtype) in cache.schema.items()}
        return ({k[2:]: v for k, v in tensors.items() if k.startswith('w.')},
                {k[2:]: v for k, v in tensors.items() if k.startswith('s.')})

    def test_roundtrip_and_precision_isolation(self):
        for mode in ('q4_0', 'q8_0'):
            cache = self.cache(mode)
            self.assertIsNone(cache.load())
            weights, scales = self.values(cache)
            self.assertTrue(cache.save(weights, scales))
            actual = self.cache(mode).load()
            for expected, loaded in zip((weights, scales), actual):
                self.assertEqual(set(expected), set(loaded))
                for key in expected:
                    self.assertTrue(torch.equal(expected[key], loaded[key]))
        self.assertNotEqual(self.cache('q4_0').key, self.cache('q8_0').key)

    def test_source_changes_invalidate_and_reject_inflight_save(self):
        cache = self.cache()
        self.source.write_bytes(b'changed source')
        self.assertNotEqual(cache.key, self.cache().key)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint changed'):
            cache.save(*self.values(cache))

    def test_config_and_format_version_invalidate(self):
        key = self.cache().key
        self.config['intermediate_size'] = 96
        self.assertNotEqual(key, self.cache().key)
        with patch('damendaemon.weight_cache.VERSION', 999):
            self.assertNotEqual(self.cache().key, key)

    def test_corruption_and_wrong_tensor_layout_are_misses(self):
        cache = self.cache()
        cache.root.mkdir()
        cache.path.write_bytes(b'truncated safetensors')
        self.assertIsNone(cache.load())
        save_file({'wrong': torch.zeros(1)}, str(cache.path), metadata={'manifest': cache.manifest})
        self.assertIsNone(cache.load())
        weights, scales = self.values(cache)
        tensors = {'w.' + k: v for k, v in weights.items()}
        tensors.update({'s.' + k: v for k, v in scales.items()})
        tensors['w.embed'] = tensors['w.embed'].float()
        save_file(tensors, str(cache.path), metadata={'manifest': cache.manifest})
        self.assertIsNone(cache.load())

    def test_failed_publish_preserves_previous_cache_and_cleans_temp(self):
        cache = self.cache()
        values = self.values(cache)
        cache.save(*values)
        before = cache.path.read_bytes()
        with patch('damendaemon.weight_cache.os.replace', side_effect=OSError('disk failure')):
            with self.assertWarns(RuntimeWarning):
                self.assertFalse(cache.save(*values))
        self.assertEqual(before, cache.path.read_bytes())
        self.assertEqual([cache.path], list(cache.root.iterdir()))

    def test_schema_covers_dense_architectures(self):
        for architecture in ('qwen2', 'qwen3', 'llama', 'phi3', 'mistral'):
            config = dict(self.config, model_type=architecture)
            schema = tensor_schema(config, 'q4_0')
            self.assertEqual('w.0.qkv.bias' in schema, architecture == 'qwen2')
            self.assertEqual('w.0.q_norm' in schema, architecture == 'qwen3')


if __name__ == '__main__':
    unittest.main()

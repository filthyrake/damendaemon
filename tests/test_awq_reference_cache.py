import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from scripts.verify_awq_cached_reference import load_oracle


class AwqReferenceCacheTests(unittest.TestCase):
    def fixture(self, directory, *, cases=None, dtype=torch.float32):
        cache, report = Path(directory)/'oracle.safetensors', Path(directory)/'report.json'
        fingerprints = {'model.safetensors': 'weights', 'config.json': 'config'}
        metadata = dict(version=1, checkpoint=fingerprints, context=512,
            cases=list(range(8)) if cases is None else cases, steps=8, prompts=[[2]]*8,
            reference_dtype='float32', teacher_forcing='independent reference greedy tokens')
        save_file({f'case_{i}_step_{j}': torch.ones(1, 32, dtype=dtype)
                   for i in range(8) for j in range(8)}, str(cache),
                  metadata={'qualification': json.dumps(metadata)})
        report.write_text(json.dumps(dict(reference_cache=dict(sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
                                         checkpoint_sha256={'model.safetensors': 'weights'})))
        return cache, report, fingerprints

    def test_exact_cache_and_checkpoint_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, report, fingerprints = self.fixture(directory)
            with patch('scripts.verify_awq_cached_reference.fingerprint', return_value=fingerprints), \
                 patch('scripts.verify_awq_cached_reference.config', return_value={'vocab_size': 32}):
                metadata, targets, _ = load_oracle(cache, report, directory)
                self.assertEqual(len(targets), 64)
                self.assertEqual(metadata['context'], 512)
            with patch('scripts.verify_awq_cached_reference.fingerprint', return_value={'model.safetensors': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'checkpoint or tokenizer differs'):
                    load_oracle(cache, report, directory)
            cache.write_bytes(cache.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'cache differs'):
                load_oracle(cache, report, directory)

    def test_partial_or_wrong_precision_oracle_is_rejected(self):
        for cases, dtype, error in [([7], torch.float32, 'complete independent'),
                                     (None, torch.float16, 'invalid oracle logit')]:
            with self.subTest(cases=cases, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                cache, report, fingerprints = self.fixture(directory, cases=cases, dtype=dtype)
                with patch('scripts.verify_awq_cached_reference.fingerprint', return_value=fingerprints), \
                     patch('scripts.verify_awq_cached_reference.config', return_value={'vocab_size': 32}):
                    with self.assertRaisesRegex(ValueError, error):
                        load_oracle(cache, report, directory)

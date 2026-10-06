"""Offline admission planning arithmetic, not GPU capacity qualification."""
import gzip
import json
from pathlib import Path
import unittest

from scripts.experiments.plan_hybrid_capacity import estimate


class HybridCapacityTests(unittest.TestCase):
    def test_precision_changes_cache_only(self):
        evidence = Path(__file__).resolve().parents[1] / 'docs/evidence/amd'
        inventory = json.loads((evidence / 'qwen38-27b-inventory.json').read_text())
        with gzip.open(evidence / 'qwen38-27b-safetensors-metadata.json.gz', 'rt') as source:
            metadata = json.load(source)
        original = json.loads((evidence / 'qwen38-27b-capacity.json').read_text())
        half = estimate(inventory, metadata)
        full = estimate(inventory, metadata, 'float32')
        self.assertEqual(half['contexts'], original['contexts'])
        self.assertEqual(half['weight_bytes'], original['weight_bytes'])
        self.assertEqual(full['weight_bytes'], half['weight_bytes'])
        # Pinned 27B geometry: 48 recurrent layers, 48 value heads, 128x128
        # state matrices. Those are already FP32 in both precision policies.
        recurrent_per_slot = 48 * 48 * 128 * 128 * 4
        for context, rows in full['contexts'].items():
            for users, values in rows.items():
                with self.subTest(context=context, users=users):
                    expected = (2 * half['contexts'][context][users]['state_bytes']
                                - int(users) * recurrent_per_slot)
                    self.assertEqual(values['state_bytes'], expected)
                    for mode, size in full['weight_bytes'].items():
                        self.assertEqual(values['weights_plus_state_bytes'][mode], size + expected)

    def test_invalid_precision_rejected_before_metadata(self):
        with self.assertRaises(ValueError):
            estimate({}, {}, 'int8')

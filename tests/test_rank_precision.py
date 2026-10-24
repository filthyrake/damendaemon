"""Per-rank heterogeneous precision (#601): pricing rules, planner, contract.

CPU-only and synthetic: packed q8_0/q4_0 weights are built with the real
quantization.prepare, the rule-priced stage_budget_formats must equal the
tensor-accounted stage_budget for every range in every format from every
extraction format, the mixed planner must stay deterministic, the chain
contract digest must pin the shared fields, and chain-serve admission must
stay honest about Q8 and uniform lists. Real-hardware mixed cells are
evidence, not tests (docs/evidence/distributed/).
"""
import copy
import unittest
from pathlib import Path

import torch

from damendaemon.chain_server_cli import main
from damendaemon.hybrid_layout import (STAGE_FORMATS, choose_stage_splits,
                                       choose_stage_splits_formats,
                                       logical_shapes, stage_budget,
                                       stage_budget_formats,
                                       stage_pricing_table)
from damendaemon.quantization import prepare
from damendaemon.stage_identity import chain_contract

CONTEXT, BATCH, CACHE_BYTES = 64, 1, 2
CONFIG = dict(num_hidden_layers=4, layer_types=['full_attention'] * 4,
              hidden_size=64, num_key_value_heads=2, head_dim=32,
              linear_num_key_heads=2, linear_num_value_heads=2,
              linear_key_head_dim=16, linear_value_head_dim=16,
              linear_conv_kernel_dim=4,
              vocab_size=128, tie_word_embeddings=True,
              max_position_embeddings=2048)


def raw_weights():
    """A tiny tied hybrid model: 4 layers, one 32-aligned matrix, a norm,
    a 3D conv state, an embedding shared with the output head."""
    weights = {'embed': torch.zeros(CONFIG['vocab_size'], CONFIG['hidden_size'],
                                    dtype=torch.float16)}
    weights['head'] = weights['embed']  # tied
    weights['norm'] = torch.zeros(CONFIG['hidden_size'], dtype=torch.float32)
    for index in range(CONFIG['num_hidden_layers']):
        weights[f'{index}.attn'] = torch.randn(64, 64, dtype=torch.float16)
        weights[f'{index}.conv'] = torch.zeros(2, 16, 4, dtype=torch.float16)
    return weights


def load_format(fmt):
    weights = raw_weights()
    scales = prepare(weights, None if fmt == 'none' else fmt)
    return weights, scales


class LogicalShapeTests(unittest.TestCase):
    def test_packed_shapes_resolve_to_logical(self):
        for fmt in ('q8_0', 'q4_0'):
            weights, scales = load_format(fmt)
            shapes = logical_shapes(weights, scales)
            self.assertEqual(shapes['0.attn'], (64, 64))
            self.assertEqual(shapes['embed'], (CONFIG['vocab_size'], 64))
            self.assertEqual(shapes['0.conv'], (2, 16, 4))

    def test_scale_disagreement_fails_closed(self):
        weights, scales = load_format('q8_0')
        tampered = dict(scales)
        tampered['0.attn'] = scales['0.attn'][:, :1]
        with self.assertRaisesRegex(ValueError, 'disagrees with its scales'):
            logical_shapes(weights, tampered)

    def test_pricing_table_is_format_independent(self):
        tables = {}
        for fmt in STAGE_FORMATS:
            weights, scales = load_format(fmt)
            tables[fmt] = stage_pricing_table(weights, scales, CONFIG)
        self.assertEqual(tables['none'], tables['q8_0'])
        self.assertEqual(tables['q8_0'], tables['q4_0'])
        # The tied embedding/head pair is canonical even though a packed
        # load rebinds the head to its own tensor.
        self.assertIn(('embed', 'head'), tables['q4_0']['shared_storage'])
        self.assertIn(('embed', 'head'), tables['none']['shared_storage'])


class StageBudgetFormatTests(unittest.TestCase):
    """Rule pricing must equal tensor accounting, cross-format, per range."""

    @classmethod
    def setUpClass(cls):
        cls.loads = {fmt: load_format(fmt) for fmt in STAGE_FORMATS}
        cls.tables = {fmt: stage_pricing_table(*cls.loads[fmt], CONFIG)
                      for fmt in STAGE_FORMATS}

    def stage_budget(self, fmt, start, end):
        weights, scales = self.loads[fmt]
        return stage_budget(weights, CONFIG, CONTEXT, BATCH, CACHE_BYTES,
                            start, end, scales=scales)

    def test_every_extraction_prices_every_target_exactly(self):
        for extract in STAGE_FORMATS:
            for target in STAGE_FORMATS:
                for start in range(CONFIG['num_hidden_layers']):
                    for end in range(start + 1, CONFIG['num_hidden_layers'] + 1):
                        actual = self.stage_budget(target, start, end)
                        predicted = stage_budget_formats(
                            self.tables[extract], CONFIG, CONTEXT, BATCH,
                            CACHE_BYTES, start, end, target)
                        for field in ('weight_bytes', 'scale_bytes', 'state_bytes',
                                      'output_head_workspace_bytes',
                                      'required_memory_bytes'):
                            self.assertEqual(predicted[field], actual[field],
                                             f'{extract}->{target} {start}:{end} {field}')

    def test_tied_head_only_rank_pays_for_the_shared_tensor(self):
        # A rank owning layers 1..4 + head holds the whole tied embedding
        # storage; only a rank owning BOTH embed and head dedups under none.
        actual = self.stage_budget('none', 1, 4)
        predicted = stage_budget_formats(self.tables['none'], CONFIG, CONTEXT,
                                         BATCH, CACHE_BYTES, 1, 4, 'none')
        self.assertEqual(predicted['weight_bytes'], actual['weight_bytes'])

    def test_unknown_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown per-rank weight format'):
            stage_budget_formats(self.tables['none'], CONFIG, CONTEXT, BATCH,
                                 CACHE_BYTES, 0, 4, 'fp8')


class MixedPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights, scales = load_format('q8_0')
        cls.table = stage_pricing_table(weights, scales, CONFIG)
        cls.weights, cls.scales = weights, scales

    def plan(self, budgets, formats):
        return choose_stage_splits_formats(self.table, CONFIG, CONTEXT, BATCH,
                                           CACHE_BYTES, budgets, formats)

    def test_uniform_list_matches_the_tensor_planner(self):
        budgets = [1024 * 1024**2] * 3
        tensor = choose_stage_splits(self.weights, CONFIG, CONTEXT, BATCH,
                                     CACHE_BYTES, budgets, scales=self.scales)
        by_formats = self.plan(budgets, ['q8_0'] * 3)
        self.assertTrue(tensor['storage_plan_found'] and by_formats['storage_plan_found'])
        self.assertEqual(tensor['splits'], by_formats['splits'])

    def test_mixed_split_prices_each_rank_in_its_format(self):
        budgets = [1024 * 1024**2, 768 * 1024**2, 2048 * 1024**2]
        mixed = self.plan(budgets, ['q8_0', 'q4_0', 'none'])
        self.assertTrue(mixed['storage_plan_found'])
        self.assertEqual(mixed['per_rank_formats'], ['q8_0', 'q4_0', 'none'])
        self.assertEqual([s['weight_format'] for s in mixed['stages']],
                         ['q8_0', 'q4_0', 'none'])
        # The same split from every extraction format: deterministic.
        for fmt in ('none', 'q4_0'):
            other = stage_pricing_table(*load_format(fmt), CONFIG)
            same = choose_stage_splits_formats(other, CONFIG, CONTEXT, BATCH,
                                               CACHE_BYTES, budgets,
                                               ['q8_0', 'q4_0', 'none'])
            self.assertEqual(same['splits'], mixed['splits'])

    def test_no_fit_reports_the_per_rank_reason(self):
        plan = self.plan([256 * 1024**2] * 3, ['none', 'none', 'none'])
        self.assertFalse(plan['storage_plan_found'])
        self.assertEqual(plan['per_rank_formats'], ['none', 'none', 'none'])
        self.assertIn('per-rank formats', plan['reason'])

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.plan([1024] * 3, ['q8_0', 'q4_0'])            # length mismatch
        with self.assertRaises(ValueError):
            self.plan([1024] * 3, ['q8_0', 'q4_0', 'fp8'])     # unknown format
        with self.assertRaises(ValueError):
            self.plan([0, 1024, 1024], ['none'] * 3)           # bad budget


class CheckpointManifestTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.owner = tempfile.TemporaryDirectory()
        self.addCleanup(self.owner.cleanup)
        self.checkpoint = Path(self.owner.name)

    def test_identical_copies_derive_the_identical_digest(self):
        from damendaemon.checkpoint_manifest import manifest_digest
        (self.checkpoint / 'crc32.txt').write_bytes(b'shard-a 1234\nshard-b 5678\n')
        first = manifest_digest(self.checkpoint)
        copy = Path(self.owner.name) / 'copy'
        copy.mkdir()
        (copy / 'crc32.txt').write_bytes(b'shard-a 1234\nshard-b 5678\n')
        self.assertEqual(first, manifest_digest(copy))
        self.assertEqual(len(first), 32)

    def test_wrong_checkpoint_changes_the_digest(self):
        from damendaemon.checkpoint_manifest import manifest_digest
        (self.checkpoint / 'crc32.txt').write_bytes(b'shard-a 1234\n')
        wrong = Path(self.owner.name) / 'wrong'
        wrong.mkdir()
        (wrong / 'crc32.txt').write_bytes(b'shard-a 9999\n')
        self.assertNotEqual(manifest_digest(self.checkpoint),
                             manifest_digest(wrong))

    def test_manifest_absent_yields_none(self):
        from damendaemon.checkpoint_manifest import manifest_digest
        self.assertIsNone(manifest_digest(self.checkpoint))


class ChainContractTests(unittest.TestCase):
    def test_same_inputs_same_digest(self):
        first = chain_contract(CONFIG, 512, ['q4_0', 'q4_0', 'none'])
        second = chain_contract(dict(CONFIG), 512, ['q4_0', 'q4_0', 'none'])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_any_shared_field_changes_the_digest(self):
        base = chain_contract(CONFIG, 512, ['q4_0', 'q4_0', 'none'])
        self.assertNotEqual(base, chain_contract(CONFIG, 512, ['q4_0', 'none', 'none']))
        self.assertNotEqual(base, chain_contract(CONFIG, 2048, ['q4_0', 'q4_0', 'none']))
        self.assertNotEqual(base, chain_contract(dict(CONFIG, hidden_size=128), 512,
                                                 ['q4_0', 'q4_0', 'none']))
        self.assertNotEqual(base, chain_contract(CONFIG, 512, ['q4_0', 'q4_0', 'none'],
                                                  boundary_dtype='float16'))

    def test_checkpoint_digest_is_pinned(self):
        # Without a checkpoint digest the contract cannot detect a wrong
        # checkpoint; with one, a different manifest changes the digest.
        digest_a = bytes(range(32))
        digest_b = bytes(range(31)) + bytes([255])
        base = chain_contract(CONFIG, 512, ['q4_0', 'q4_0', 'none'])
        with_a = chain_contract(CONFIG, 512, ['q4_0', 'q4_0', 'none'],
                                checkpoint_digest=digest_a)
        with_b = chain_contract(CONFIG, 512, ['q4_0', 'q4_0', 'none'],
                                checkpoint_digest=digest_b)
        self.assertNotEqual(base, with_a)
        self.assertNotEqual(with_a, with_b)

    def test_validation(self):
        with self.assertRaises(ValueError):
            chain_contract(CONFIG, 512, ['q4_0'])               # too few ranks
        with self.assertRaises(ValueError):
            chain_contract(CONFIG, 0, ['q4_0', 'none'])           # bad context
        with self.assertRaises(ValueError):
            chain_contract(CONFIG, 512, ['', 'none'])             # empty name
        with self.assertRaises(ValueError):
            chain_contract(CONFIG, 512, [None, 'none'])           # not strings
        with self.assertRaises(ValueError):
            chain_contract(CONFIG, 512, ['q4_0', 'none'],
                           checkpoint_digest=b'too-short')        # bad digest length


class MixedChainAdmissionTests(unittest.TestCase):
    def error(self, argv, world=3):
        import os
        from unittest.mock import patch
        with patch.dict('os.environ', {'RANK': '0', 'WORLD_SIZE': str(world),
                                       'LOCAL_WORLD_SIZE': '2'}, clear=True):
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        return caught.exception

    def test_q8_participation_stays_rejected_with_policy_reason(self):
        failure = self.error(['model', '--quantization', 'q4_0,q8_0,none',
                              '--rank-budgets-mib', '22528,13312,22528'])
        self.assertEqual(failure.code, 2)

    def test_uniform_list_is_rejected(self):
        self.error(['model', '--quantization', 'q4_0,q4_0,q4_0',
                    '--rank-budgets-mib', '22528,13312,22528'])

    def test_mixed_requires_rank_budgets(self):
        self.error(['model', '--quantization', 'q4_0,q4_0,none'])

    def test_mixed_context_bound_is_the_qualified_chain_cell(self):
        self.error(['model', '--quantization', 'q4_0,q4_0,none',
                    '--rank-budgets-mib', '22528,13312,22528', '--context', '2049'])

    def test_mixed_needs_one_entry_per_rank(self):
        failure = self.error(['model', '--quantization', 'q4_0,none',
                              '--rank-budgets-mib', '22528,13312,22528'])
        self.assertIn('exactly 3 entries', str(failure))

    def test_mixed_reaches_the_runtime_with_the_qualified_shape(self):
        # Full validation passes; the budget-count guard fires at runtime.
        failure = self.error(['model', '--quantization', 'q4_0,q4_0,none',
                              '--rank-budgets-mib', '22528,13312',
                              '--context', '2048'])
        self.assertIn('exactly 3 entries', str(failure))


if __name__ == '__main__':
    unittest.main()

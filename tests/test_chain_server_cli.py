"""chain-serve CLI admission for the quantized chain path (#536 gate 4).

CPU-only: argument validation and the node-local admission-budget slice.
The quantized serving path itself is qualified on the real chain (see
docs/evidence/distributed/qwen38-gate4-plan-536/).
"""
import unittest
from unittest.mock import patch

from damendaemon.chain_server_cli import main, node_local_budgets


class NodeLocalBudgetsTests(unittest.TestCase):
    def test_slice_follows_the_rank_node(self):
        # labnode3 ranks 0/1 see both GPUs; the desktop rank 2 sees one.
        budgets = [22528, 13312, 22528]
        self.assertEqual(node_local_budgets(0, 0, 2, budgets), [22528, 13312])
        self.assertEqual(node_local_budgets(1, 1, 2, budgets), [22528, 13312])
        self.assertEqual(node_local_budgets(2, 0, 1, budgets), [22528])

    def test_rejects_slice_outside_the_budgets(self):
        with self.assertRaises(ValueError):
            node_local_budgets(0, 0, 2, [22528])  # node larger than the list
        with self.assertRaises(ValueError):
            node_local_budgets(0, 1, 1, [22528, 13312])  # local rank above the rank
        with self.assertRaises(ValueError):
            node_local_budgets(0, 0, 0, [22528])  # empty node


class ChainServeArgTests(unittest.TestCase):
    def error(self, argv):
        with patch.dict('os.environ', {'RANK': '0', 'WORLD_SIZE': '3'}, clear=True):
            with self.assertRaises(SystemExit) as caught:
                main(argv)
        self.assertEqual(caught.exception.code, 2)

    def test_q4_requires_rank_budgets(self):
        self.error(['model', '--quantization', 'q4_0', '--context', '2048'])

    def test_q4_context_bound_is_the_qualified_cell(self):
        self.error(['model', '--quantization', 'q4_0',
                    '--rank-budgets-mib', '22528,13312,22528',
                    '--context', '2049'])

    def test_unquantized_context_stays_at_512(self):
        self.error(['model', '--context', '513'])

    def test_q8_chain_not_qualified(self):
        self.error(['model', '--quantization', 'q8_0', '--context', '512'])

    def test_rank_budgets_must_be_positive_integers(self):
        self.error(['model', '--quantization', 'q4_0',
                    '--rank-budgets-mib', '22528,-13312,22528',
                    '--context', '2048'])

    def test_chunk_size_must_fit_context(self):
        self.error(['model', '--quantization', 'q4_0',
                    '--rank-budgets-mib', '22528,13312,22528',
                    '--context', '2048', '--chunk-size', '4097'])

    def test_budget_count_must_match_world(self):
        # WORLD_SIZE=3 but only two budgets: fails AFTER parsing (the torch
        # environment is present), so SystemExit carries the mismatch text.
        with patch.dict('os.environ', {'RANK': '0', 'WORLD_SIZE': '3',
                                       'LOCAL_WORLD_SIZE': '2'}, clear=True):
            with self.assertRaises(SystemExit) as caught:
                main(['model', '--quantization', 'q4_0',
                      '--rank-budgets-mib', '22528,13312', '--context', '2048'])
        self.assertIn('exactly 3 entries', str(caught.exception))


if __name__ == '__main__':
    unittest.main()

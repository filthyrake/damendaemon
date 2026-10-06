"""Offline placement optimality and rejection; never hardware fit evidence."""
from fractions import Fraction
from itertools import combinations
import unittest

from dllm.hybrid_layout import choose_stage_splits, stage_budget
from tests import test_hybrid_stage


class HybridPlacementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_hybrid_stage.HybridStageTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model

    def choose(self, budgets):
        return choose_stage_splits(self.model.weights, self.model.config, 32, 2, 4, budgets, True)

    def test_matches_exhaustive_partitions_for_heterogeneous_budgets(self):
        for budgets in ([2**30, 2**30], [550_000_000, 600_000_000],
                        [600_000_000, 550_000_000], [2**30]*3, [2**30]*4):
            with self.subTest(budgets=budgets):
                scores = []
                for split in combinations(range(1, 4), len(budgets)-1):
                    edges = (0, *split, 4)
                    required = [stage_budget(self.model.weights, self.model.config, 32, 2, 4, start, end, True)
                                ['required_memory_bytes'] for start, end in zip(edges, edges[1:])]
                    if all(size <= limit for size, limit in zip(required, budgets)):
                        scores.append(max(Fraction(size, limit) for size, limit in zip(required, budgets)))
                result = self.choose(budgets)
                self.assertEqual(result['storage_plan_found'], bool(scores))
                if scores:
                    self.assertEqual(result['maximum_estimated_utilization'], float(min(scores)))
                    self.assertEqual(len(result['stages']), len(budgets))
                    self.assertTrue(all(stage['remaining_budget_bytes'] >= 0 for stage in result['stages']))
                    self.assertEqual(result, self.choose(budgets))

    def test_no_fit_preserves_requested_limits(self):
        result = self.choose([1, 2])
        self.assertFalse(result['storage_plan_found'])
        self.assertEqual(result['device_budget_bytes'], [1, 2])
        self.assertNotIn('splits', result)

    def test_exact_fit_and_one_byte_short(self):
        required = stage_budget(self.model.weights, self.model.config, 32, 2, 4, 0, 4, True)['required_memory_bytes']
        self.assertTrue(self.choose([required])['storage_plan_found'])
        self.assertFalse(self.choose([required-1])['storage_plan_found'])

    def test_invalid_budgets_rejected(self):
        for budgets in ([], [0], [-1], [True], [1.5], [2**30]*5):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                self.choose(budgets)

"""CPU property tests for the #687 quality battery's scoring and corpus
construction (#687). Pure python: a stub tokenizer (word-count ids) drives
the corpus assembler; synthetic answers drive the parsers/scorers. No GPU
execution or model claim — the pinned contract is: budgets respected,
needle placement deterministic, scoring exact-agreement."""
import unittest
from unittest.mock import patch

from scripts.experiments.long_context_quality_battery_687 import (
    GATES, T1_NEEDLES, T2_NEEDLES, T3_TRUTH, T4_KEYS, QA_QUESTIONS,
    assemble, normalize, score_qa, score_summary)


class StubEncoding:
    def __init__(self, ids):
        self.ids = ids


class StubTokenizer:
    """One id per whitespace-separated word — deterministic, budget-safe."""

    def encode(self, text):
        return StubEncoding(text.split())


class QualityBatteryTests(unittest.TestCase):
    def test_assemble_respects_budget_and_places_needles(self):
        tokenizer = StubTokenizer()
        inserts = [('t1', 'NEEDLE ONE amber K7Q-4471.', 0.25),
                   ('t2', 'NEEDLE TWO Halden 47.', 0.75)]
        prompt, layout, corpus_tokens, prompt_tokens = assemble(
            tokenizer, 500, inserts, QA_QUESTIONS)
        self.assertLessEqual(prompt_tokens, 500 + 700)
        kinds = [kind for kind, _, _ in layout if kind != 'filler']
        self.assertEqual(kinds, ['t1', 't2'], 'order must follow fractions')
        # The measured fractions must land near the targets.
        positions = {kind: position for kind, _, position in layout if kind != 'filler'}
        self.assertLess(abs(positions['t1'] / max(corpus_tokens, 1) - 0.25), 0.15)
        self.assertLess(abs(positions['t2'] / max(corpus_tokens, 1) - 0.75), 0.15)
        # Determinism: identical inputs assemble byte-identically.
        again = assemble(tokenizer, 500, inserts, QA_QUESTIONS)
        self.assertEqual(prompt, again[0])

    def test_assemble_scales_to_budget(self):
        tokenizer = StubTokenizer()
        small = assemble(tokenizer, 300, [], 'end')
        large = assemble(tokenizer, 3000, [], 'end')
        self.assertLess(small[3], large[3])
        self.assertGreater(large[3], 2000)

    def test_score_qa_exact_and_format_tolerant(self):
        output = ('\nA1: K7Q-4471\nA2: The code is RX3-9028.\nA3: k7q4471\n'
                  'A4: Halden 47; Ostmark: 52; Vireo 38; Casselt 61; Dunmore 44; Pellworth 57\n'
                  'A5: YES\nA6: No.')
        answers, tasks = score_qa(output, [])
        self.assertEqual(answers['A1'], 'K7Q-4471')
        # T1: amber exact; cobalt embedded in prose (still exact); mauve
        # wrong (parrots the amber code) — 2/3, the declared gate.
        self.assertEqual([item['passed'] for item in tasks['T1']], [True, True, False])
        self.assertTrue(all(item['passed'] for item in tasks['T2']))
        self.assertEqual([item['passed'] for item in tasks['T3']], [True, True])

    def test_score_qa_missing_and_garbled(self):
        output = 'A1: unknown\nA2: \nA3: V8N-6113'
        _, tasks = score_qa(output, [])
        self.assertEqual([item['passed'] for item in tasks['T1']], [False, False, True])
        self.assertFalse(any(item['passed'] for item in tasks['T2']))
        self.assertFalse(any(item['passed'] for item in tasks['T3']))

    def test_score_summary_keys(self):
        text = ('A thermal runaway struck tower 4; crew Halvard contained it and '
                'deck 7 was evacuated in 48 minutes.')
        items = score_summary(text)
        self.assertTrue(all(item['passed'] for item in items), items)
        partial = score_summary('A thermal runaway was contained.')
        self.assertEqual(sum(1 for item in partial if item['passed']), 1)
        self.assertEqual(len(partial), len(T4_KEYS))

    def test_normalize_and_gate_constants_pinned(self):
        self.assertEqual(normalize('K7Q-4471, Yes!'), 'k7q 4471 yes')
        # The declared gate minimums are part of the pinned contract.
        self.assertEqual(GATES, dict(t1_min=2, t1_total=3, t2_min=4, t2_total=6,
                                     t3_min=1, t3_total=2, t4_min=3, t4_total=5))
        self.assertEqual(T3_TRUTH, {'Q5': 'yes', 'Q6': 'no'})
        self.assertEqual(len(T1_NEEDLES), 3)
        self.assertEqual(len(T2_NEEDLES), 6)
        self.assertEqual(len(T4_KEYS), 5)


if __name__ == '__main__':
    unittest.main()

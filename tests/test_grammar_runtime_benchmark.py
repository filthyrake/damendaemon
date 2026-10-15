import unittest
from types import SimpleNamespace
from scripts.benchmark_grammar_runtime import measure


class GrammarRuntimeBenchmarkTests(unittest.TestCase):
    def test_pairs_alternate_and_tokens_match(self):
        class Constraint:
            def __init__(self,*args):pass
            def apply(self,logits):return logits.clone()
            def accept(self,token):pass
        report=measure(SimpleNamespace(config={'vocab_size':4}),[1,2],pairs=2,constraint=Constraint)
        self.assertEqual([r['arm'] for r in report['samples']],['plain']*2+['constrained']*4+['plain']*2)
        self.assertEqual(len(report['paired_extra_seconds_per_token']),2)
        self.assertFalse(report['end_to_end_performance_qualified'])

    def test_mutating_and_disallowing_constraints_fail(self):
        class Constraint:
            def __init__(self,*args):pass
            def apply(self,logits):logits.fill_(0);return logits
            def accept(self,token):pass
        model=SimpleNamespace(config={'vocab_size':4})
        with self.assertRaisesRegex(ValueError,'mutated'):measure(model,[1],pairs=2,constraint=Constraint)
        class Disallow(Constraint):
            def apply(self,logits):return logits.clone().fill_(float('-inf'))
        with self.assertRaisesRegex(ValueError,'not permitted'):measure(model,[1],pairs=2,constraint=Disallow)

    def test_bounds(self):
        model=SimpleNamespace(config={'vocab_size':4})
        for tokens,pairs in [([],2),([1],1),([1],21),([4],2),([True],2)]:
            with self.assertRaises(ValueError):measure(model,tokens,pairs)


if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace
from scripts.benchmark_grammar_setup import benchmark


class GrammarSetupBenchmarkTests(unittest.TestCase):
    def test_fresh_models_warm_reuse_and_distinct_schemas(self):
        models=[];calls=[]
        def factory():
            model=SimpleNamespace();models.append(model);return model
        def constraint(model,schema):calls.append((model,schema))
        report=benchmark(factory,2,3,constraint)
        self.assertEqual(len(models),2)
        self.assertEqual(len(calls),14)
        self.assertEqual(len(report['samples']),20)
        for group in (calls[:7],calls[7:]):
            self.assertTrue(all(m is group[0][0] for m,s in group))
            self.assertEqual([s for m,s in group][::2][1:], [group[2][1],group[4][1],group[6][1]])
            self.assertEqual(len({s for m,s in group}),4)
        self.assertFalse(report['gpu_performance_qualified'])
        self.assertFalse(report['model_weights_loaded'])

    def test_bounds_and_invalid_timing(self):
        for trials,repeats in [(0,1),(6,1),(1,11),(True,1)]:
            with self.assertRaises(ValueError):benchmark(lambda:None,trials,repeats,lambda *a:None)
        with self.assertRaises(ValueError):benchmark(lambda:None,1,1,lambda *a:None,lambda:float('nan'))


if __name__=='__main__':unittest.main()

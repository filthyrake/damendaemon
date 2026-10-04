import math
import unittest
import torch
from dllm.sampling import Sampling, Sampler, sample_rows


class SamplingTests(unittest.TestCase):
    def test_validation(self):
        for options in ({'temperature':True}, {'temperature':float('inf')},
                        {'temperature':-1}, {'top_p':0}, {'top_p':float('nan')},
                        {'top_k':False}, {'top_k':-1}, {'seed':2**64}, {'seed':1.2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Sampling(**options)

    def test_greedy_and_top_one(self):
        logits = torch.tensor([-.1, 2., .7])
        for options in (Sampling(), Sampling(1, top_k=1), Sampling(1, top_p=.01)):
            self.assertEqual(Sampler(options).select(logits).item(), 1)

    def test_seed_stream_is_independent_of_batch_order(self):
        a, b = Sampler(Sampling(.8, seed=42)), Sampler(Sampling(.8, seed=42))
        other = Sampler(Sampling(1, seed=1))
        logits = torch.tensor([[.1, .8, -.2], [.1, .8, -.2]])
        for _ in range(50):
            expected = a.select(logits[0]).item()
            actual = sample_rows(logits, [other, b])[1]
            self.assertEqual(actual, expected)

    def test_distribution_and_top_k_filter(self):
        logits = torch.tensor([math.log(.2), math.log(.3), math.log(.5)])
        sampler = Sampler(Sampling(1, seed=42))
        counts = [0, 0, 0]
        for _ in range(4000):
            counts[sampler.select(logits).item()] += 1
        for count, target in zip(counts, (.2, .3, .5)):
            self.assertAlmostEqual(count / 4000, target, delta=.03)
        sampler = Sampler(Sampling(1, top_k=2, seed=42))
        self.assertNotIn(0, {sampler.select(logits).item() for _ in range(100)})

    def test_nucleus_keeps_crossing_token(self):
        logits = torch.tensor([math.log(.5), math.log(.3), math.log(.2)])
        sampler = Sampler(Sampling(1, top_p=.6, seed=42))
        self.assertEqual({sampler.select(logits).item() for _ in range(100)}, {0, 1})

    def test_small_temperature_and_invalid_logits(self):
        self.assertEqual(Sampler(Sampling(1e-300)).select(torch.tensor([1.,2.])).item(), 1)
        with self.assertRaises(ValueError):
            Sampler(Sampling(1)).select(torch.tensor([float('nan'), 1.]))

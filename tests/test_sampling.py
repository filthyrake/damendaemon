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

    def test_penalties_use_correct_history_and_preserve_logits(self):
        sampler = Sampler(Sampling(presence_penalty=.5, frequency_penalty=.25,
                                  repetition_penalty=2, logit_bias={'3':1}))
        sampler.prepare([0,1,1],4)
        sampler.observe(1)
        sampler.observe(1)
        sampler.observe(2)
        logits = torch.tensor([4.,-2.,3.,0.])
        actual = sampler.adjust(logits)
        self.assertTrue(torch.equal(actual,torch.tensor([2.,-5.,.75,1.])))
        self.assertTrue(torch.equal(logits,torch.tensor([4.,-2.,3.,0.])))
        sampler.prepare([],4)
        self.assertTrue(torch.equal(sampler.adjust(logits),torch.tensor([4.,-2.,3.,1.])))

    def test_penalty_validation_and_bias_snapshot(self):
        for options in ({'presence_penalty':3}, {'frequency_penalty':-3},
                        {'repetition_penalty':0}, {'repetition_penalty':True},
                        {'presence_penalty':float('nan')}, {'logit_bias':[]},
                        {'logit_bias':{'01':1}}, {'logit_bias':{'-1':1}},
                        {'logit_bias':{True:1}}, {'logit_bias':{'1':101}},
                        {'logit_bias':{'1':float('inf')}}, {'logit_bias':{'1':True}},
                        {'logit_bias':{'1':1,1:2}}, {'logit_bias':dict.fromkeys(range(1025),1)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Sampling(**options)
        bias = {'1':2}
        options = Sampling(logit_bias=bias)
        bias['1']=100
        self.assertEqual(options.logit_bias,{1:2})
        with self.assertRaises(ValueError):
            Sampler(options).prepare([],1)

    def test_penalties_disable_only_affected_greedy_shortcut(self):
        plain = Sampler(Sampling())
        adjusted = Sampler(Sampling(frequency_penalty=2))
        adjusted.prepare([0],3)
        logits = torch.tensor([[3.,2.,1.],[3.,2.,1.]])
        adjusted.observe(0)
        self.assertEqual(sample_rows(logits,[plain,adjusted],torch.tensor([0,0])),[0,1])
        self.assertIs(plain.adjust(logits[0])._base,logits)
        self.assertEqual(plain.counts,{})

    def test_presence_and_frequency_exclude_prompt(self):
        sampler = Sampler(Sampling(presence_penalty=1,frequency_penalty=-.5))
        sampler.prepare([0]*100,3)
        logits = torch.tensor([1.,2.,3.])
        self.assertTrue(torch.equal(sampler.adjust(logits),logits))
        sampler.observe(0)
        self.assertTrue(torch.equal(sampler.adjust(logits),torch.tensor([.5,2.,3.])))

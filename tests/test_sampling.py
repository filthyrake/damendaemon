import math
import unittest
from unittest.mock import patch
import torch
from dllm.sampling import Sampling, Sampler, sample_rows


class SamplingTests(unittest.TestCase):
    def test_upper_endpoint_draw_never_selects_filtered_tokens(self):
        for options, logits in (
                (Sampling(1, top_p=.5), torch.tensor([0., -10., -20.])),
                (Sampling(1), torch.tensor([0., -1000., -1000.])),
                (Sampling(1, top_k=2, top_p=.5), torch.tensor([0., -10., -20.]))):
            with self.subTest(options=options):
                sampler = Sampler(options)
                with patch.object(sampler.random, 'random', return_value=1 - 2**-30):
                    self.assertEqual(sampler.select(logits).item(), 0)
                    self.assertEqual(sample_rows(logits.unsqueeze(0), [sampler]), [0])

    def test_cdf_roundoff_does_not_assign_mass_to_zero_probability_tail(self):
        sampler = Sampler(Sampling(1))
        logits = torch.tensor([3., 2., 1., -1000.])
        with patch.object(sampler.random, 'random', return_value=1 - 2**-24):
            self.assertEqual(sampler.select(logits).item(), 2)


    def test_malformed_selected_batches_are_runtime_failures(self):
        logits = torch.zeros(2, 4)
        samplers = [Sampler(Sampling()), Sampler(Sampling())]
        for selected in (torch.tensor([]), torch.tensor([1]), torch.tensor([1,2,3]),
                         torch.tensor([[1],[2]]), torch.tensor([1.,2.]),
                         torch.tensor([True,False]), torch.tensor([-1,2])):
            with self.subTest(selected=selected), self.assertRaises(RuntimeError):
                sample_rows(logits, samplers, selected)
        for sampler in samplers: sampler.logprobs_top = 0
        with patch('dllm.logprobs.score_rows', return_value=[{'token_id': 1}]):
            with self.assertRaises(RuntimeError): sample_rows(logits, samplers)

    def test_selected_ids_must_fit_vocabulary(self):
        logits = torch.zeros(2, 4)
        samplers = [Sampler(Sampling()), Sampler(Sampling())]
        for selected in ([4, 1], [1, 10**6]):
            with self.subTest(selected=selected), self.assertRaises(RuntimeError):
                sample_rows(logits, samplers, torch.tensor(selected))
        self.assertEqual(sample_rows(logits, samplers, torch.tensor([0, 3])), [0, 3])

    def test_scored_packet_bounds_and_cardinality_before_state_publication(self):
        logits = torch.zeros(2, 4)
        for all_scored in (False, True):
            samplers = [Sampler(Sampling()), Sampler(Sampling())]
            samplers[0].logprobs_top = 0
            if all_scored:
                samplers[1].logprobs_top = 0
            valid = [{'token_id':0}] * (2 if all_scored else 1)
            bad = (None, {}, [None], [{}], [], valid + [{'token_id':1}], [{'token_id':4}] + valid[1:],
                   [{'token_id':True}] + valid[1:])
            for packets in bad:
                sentinel = object()
                for sampler in samplers:
                    sampler.last_logprobs = sentinel
                with self.subTest(all_scored=all_scored, packets=packets):
                    with patch('dllm.logprobs.score_rows', return_value=packets):
                        with self.assertRaises(RuntimeError):
                            sample_rows(logits, samplers)
                    for sampler in samplers:
                        self.assertIs(sampler.last_logprobs, sentinel)

    def test_mixed_scored_ids_must_match_existing_selected_readback(self):
        samplers = [Sampler(Sampling()), Sampler(Sampling())]
        samplers[0].logprobs_top = 0
        with patch('dllm.logprobs.score_rows', return_value=[{'token_id':1}]):
            with self.assertRaisesRegex(RuntimeError, 'disagree'):
                sample_rows(torch.zeros(2, 4), samplers)
        self.assertIsNone(samplers[0].last_logprobs)

    def test_validation(self):
        for options in ({'temperature':True}, {'temperature':float('inf')},
                        {'temperature':-1}, {'top_p':0}, {'top_p':float('nan')},
                        {'top_k':False}, {'top_k':-1}, {'seed':2**64}, {'seed':1.2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Sampling(**options)

    def test_oversized_integer_options_are_validation_errors(self):
        for value in (10**400, -(10**400)):
            for name in ('temperature', 'top_p', 'presence_penalty',
                         'frequency_penalty', 'repetition_penalty', 'logit_bias'):
                options = {name: {'0': value} if name == 'logit_bias' else value}
                with self.subTest(name=name, sign=value > 0), self.assertRaises(ValueError):
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

    def test_penalty_overflow_is_rejected_before_selection(self):
        cases = ((1e-300, [1., 2.]), (1e300, [-2., -1.]), (1e300, [0., -1.]),
                 (2., [-3e38, -2e38]))
        for penalty, values in cases:
            for temperature in (0, 1):
                with self.subTest(penalty=penalty, values=values, temperature=temperature):
                    sampler = Sampler(Sampling(temperature, repetition_penalty=penalty))
                    sampler.prepare([0, 1], 2)
                    logits = torch.tensor(values)
                    original = logits.clone()
                    with self.assertRaisesRegex(ValueError, 'non-finite adjusted'):
                        sampler.select(logits)
                    self.assertTrue(torch.equal(logits, original))
                    with self.assertRaisesRegex(ValueError, 'non-finite adjusted'):
                        sample_rows(torch.stack([torch.zeros(2), logits]), [Sampler(Sampling()), sampler])
        ordinary = torch.tensor([1., 2.])
        plain = Sampler(Sampling())
        self.assertIs(plain.adjust(ordinary), ordinary)
        valid = Sampler(Sampling(repetition_penalty=2.))
        valid.prepare([0, 1], 2)
        self.assertEqual(valid.select(ordinary).item(), 1)

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

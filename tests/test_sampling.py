import math
import unittest
from unittest.mock import patch
import torch
from damendaemon.sampling import Sampling, Sampler, sample_rows


class SamplingTests(unittest.TestCase):
    def test_large_temperature_matches_float64_distribution(self):
        logits=torch.tensor([3e38,-3e38,-float('inf')])
        for temperature in (3e38,1e39,1e300):
            probabilities=torch.softmax(logits.double()/temperature,-1)
            boundary=probabilities[0].item()
            for draw,expected in ((boundary/2,0),((1+boundary)/2,1)):
                sampler=Sampler(Sampling(temperature))
                with self.subTest(temperature=temperature,draw=draw),patch.object(sampler.random,'random',return_value=draw):
                    self.assertEqual(sampler.select(logits).item(),expected)
                    self.assertEqual(sample_rows(logits.unsqueeze(0),[sampler]),[expected])

    def test_large_temperature_preserves_filtering_and_invalid_logits(self):
        logits=torch.tensor([3e38,-3e38,-float('inf')])
        for temperature in (3e38,1e39,1e300):
            sampler=Sampler(Sampling(temperature,top_k=2,top_p=.95))
            with self.subTest(temperature=temperature),patch.object(sampler.random,'random',return_value=.99):
                self.assertEqual(sampler.select(logits).item(),1)
            for values in ([0.,float('nan')],[0.,float('inf')],[-float('inf')]*2):
                with self.subTest(temperature=temperature,values=values),self.assertRaises(RuntimeError):
                    sampler.select(torch.tensor(values))

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


    def test_logit_batch_shape_is_checked_before_sampling(self):
        for options in (Sampling(), Sampling(1, seed=42),
                        Sampling(logit_bias={'0': 1})):
            for shape in ((), (4,), (1, 4), (3, 4), (2, 1, 4), (2, 0)):
                samplers = [Sampler(options), Sampler(options)]
                for sampler in samplers:
                    sampler.logprobs_top = 1
                states = [sampler.random.getstate() if sampler.random else None for sampler in samplers]
                with self.subTest(options=options, shape=shape), \
                        self.assertRaisesRegex(RuntimeError, 'sampling logit batch'):
                    sample_rows(torch.zeros(shape), samplers)
                self.assertEqual(states, [sampler.random.getstate() if sampler.random else None for sampler in samplers])
                self.assertTrue(all(s.last_logprobs is None for s in samplers))
        # Empty batches retain the ordinary empty result.
        self.assertEqual(sample_rows(torch.empty(0, 4), []), [])
        for options in (Sampling(), Sampling(1, seed=42),
                        Sampling(logit_bias={'0': 1})):
            samplers = [Sampler(options), Sampler(options)]
            self.assertEqual(sample_rows(torch.tensor([[0., -1000.], [-1000., 0.]]),
                                         samplers), [0, 1])

    def test_malformed_selected_batches_are_runtime_failures(self):
        logits = torch.zeros(2, 4)
        samplers = [Sampler(Sampling()), Sampler(Sampling())]
        for selected in (torch.tensor([]), torch.tensor([1]), torch.tensor([1,2,3]),
                         torch.tensor([[1],[2]]), torch.tensor([1.,2.]),
                         torch.tensor([True,False]), torch.tensor([-1,2])):
            with self.subTest(selected=selected), self.assertRaises(RuntimeError):
                sample_rows(logits, samplers, selected)
        for sampler in samplers: sampler.logprobs_top = 0
        with patch('damendaemon.logprobs.score_rows', return_value=[{'token_id': 1}]):
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
                    with patch('damendaemon.logprobs.score_rows', return_value=packets):
                        with self.assertRaises(RuntimeError):
                            sample_rows(logits, samplers)
                    for sampler in samplers:
                        self.assertIs(sampler.last_logprobs, sentinel)

    def test_mixed_scored_ids_must_match_existing_selected_readback(self):
        samplers = [Sampler(Sampling()), Sampler(Sampling())]
        samplers[0].logprobs_top = 0
        with patch('damendaemon.logprobs.score_rows', return_value=[{'token_id':1}]):
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
        with self.assertRaises(RuntimeError):
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

    def test_nonfinite_greedy_logits_fail_closed(self):
        for row in ([float('nan'),1.], [1.,float('nan')], [float('inf'),1.],
                    [-float('inf'),-float('inf')]):
            logits=torch.tensor(row)
            with self.subTest(row=row),self.assertRaisesRegex(RuntimeError,'non-finite greedy'):
                Sampler(Sampling()).select(logits)
            batch=torch.stack([torch.tensor([1.,0.]),logits])
            for cached in (None,torch.tensor([0,0])):
                samplers=[Sampler(Sampling()),Sampler(Sampling())]
                with self.subTest(row=row,cached=cached),self.assertRaisesRegex(RuntimeError,'invalid selected token batch'):
                    sample_rows(batch,samplers,cached)

    def test_greedy_negative_infinity_masks_and_ties(self):
        logits=torch.tensor([[-float('inf'),2.,2.],[1.,-float('inf'),0.]])
        samplers=[Sampler(Sampling()),Sampler(Sampling())]
        for cached in (None,torch.tensor([1,0])):
            self.assertEqual(sample_rows(logits,samplers,cached),[1,0])
        self.assertEqual([s.select(row).item() for s,row in zip(samplers,logits)],[1,0])

    def test_adjustments_preserve_existing_negative_infinity_masks(self):
        policies=({'presence_penalty':.5},{'frequency_penalty':.25},
                  {'repetition_penalty':2},{'logit_bias':{'0':5,'2':1}},
                  {'presence_penalty':.5,'frequency_penalty':.25,'repetition_penalty':2,
                   'logit_bias':{'0':5,'2':1}})
        logits=torch.tensor([-float('inf'),4.,1.])
        for policy in policies:
            for temperature in (0,1):
                with self.subTest(policy=policy,temperature=temperature):
                    sampler=Sampler(Sampling(temperature,seed=73,**policy))
                    sampler.prepare([0,1],3);sampler.observe(0);sampler.observe(1)
                    adjusted=sampler.adjust(logits)
                    self.assertTrue(torch.isneginf(adjusted[0]))
                    self.assertTrue(torch.isfinite(adjusted[1:]).all())
                    self.assertIn(sampler.select(logits).item(),(1,2))
                    selected=sample_rows(torch.stack([torch.zeros(3),logits]),
                        [Sampler(Sampling()),sampler],torch.tensor([0,0]))
                    self.assertEqual(selected[0],0);self.assertIn(selected[1],(1,2))
                    torch.testing.assert_close(logits,torch.tensor([-float('inf'),4.,1.]))
        torch.testing.assert_close(adjusted,torch.tensor([-float('inf'),1.25,2.]))

    def test_adjustments_do_not_make_corrupt_or_all_masked_rows_valid(self):
        for temperature in (0,1):
            for row in ([float('nan'),1.],[float('inf'),1.],[-float('inf'),-float('inf')]):
                sampler=Sampler(Sampling(temperature,logit_bias={'0':1}))
                sampler.prepare([],2)
                with self.subTest(row=row,temperature=temperature),self.assertRaisesRegex(RuntimeError,'non-finite'):
                    sampler.select(torch.tensor(row))

    def test_checked_greedy_selection_has_no_host_readback(self):
        from damendaemon.sampling import greedy_tokens
        logits=torch.tensor([[1.,2.],[float('nan'),0.]])
        with patch.object(torch.Tensor,'item',side_effect=AssertionError('host item read')), \
                patch.object(torch.Tensor,'tolist',side_effect=AssertionError('host list read')), \
                patch.object(torch.Tensor,'cpu',side_effect=AssertionError('host tensor copy')):
            ordinary=greedy_tokens(logits)
            cached=greedy_tokens(logits,torch.tensor([1,0]))
        self.assertEqual(ordinary.tolist(),[1,-1]);self.assertEqual(cached.tolist(),[1,-1])

    def test_scored_greedy_failure_never_publishes_packets(self):
        for all_scored in (False,True):
            for bad in (float('nan'),float('inf'),-float('inf')):
                samplers=[Sampler(Sampling()),Sampler(Sampling())]
                samplers[1].logprobs_top=1
                if all_scored:samplers[0].logprobs_top=1
                logits=torch.tensor([[1.,2.],[bad,bad]])
                with self.subTest(all_scored=all_scored,bad=bad),self.assertRaisesRegex(RuntimeError,'non-finite'):
                    sample_rows(logits,samplers)
                self.assertTrue(all(s.last_logprobs is None for s in samplers))

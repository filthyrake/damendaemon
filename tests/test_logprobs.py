import gc
import math
import unittest
from unittest.mock import patch
import weakref
import torch
from tokenizers import Tokenizer,decoders,models

from dllm.logprobs import TokenText,options,scores
from dllm.sampling import Sampler,Sampling,sample_rows


class LogprobTests(unittest.TestCase):
    def test_scores_match_distribution_and_include_selected_outside_top(self):
        logits=torch.tensor([3.,2.,1.,0.])
        original=logits.clone()
        result=scores(logits,3,2)
        expected=torch.log_softmax(logits.double(),-1)
        self.assertAlmostEqual(result['logprob'],expected[3].item(),places=6)
        self.assertEqual([token for token,_ in result['top']],[0,1])
        for token,value in result['top']:self.assertAlmostEqual(value,expected[token].item(),places=6)
        self.assertTrue(torch.equal(logits,original))
        self.assertEqual(scores(logits,0,0)['top'],[])

    def test_bias_selects_token_but_scores_raw_distribution(self):
        sampler=Sampler(Sampling(logit_bias={2:100}))
        sampler.logprobs_top=2
        logits=torch.tensor([4.,1.,0.]);original=logits.clone()
        self.assertEqual(sampler.select(logits).item(),2)
        self.assertAlmostEqual(sampler.last_logprobs['logprob'],torch.log_softmax(logits,0)[2].item())
        self.assertTrue(torch.equal(logits,original))

    def test_large_logit_offset_does_not_lose_probability_precision(self):
        logits=torch.tensor([1.e6,1.e6+1,1.e6+2])
        result=scores(logits,0,2)
        expected=torch.log_softmax(logits.double(),-1)
        self.assertAlmostEqual(result['logprob'],expected[0].item(),places=6)

    def test_scoring_does_not_change_seeded_sampling(self):
        plain=Sampler(Sampling(temperature=.8,top_p=.9,top_k=3,seed=73))
        scored=Sampler(plain.options);scored.logprobs_top=3
        logits=torch.tensor([1.,2.,3.,4.])
        self.assertEqual([plain.select(logits).item() for _ in range(30)],
                         [scored.select(logits).item() for _ in range(30)])

    def test_default_fast_path_does_not_score_and_requests_are_isolated(self):
        left,right=Sampler(Sampling()),Sampler(Sampling())
        logits=torch.tensor([[3.,1.],[0.,2.]])
        with patch('dllm.logprobs.scores',side_effect=AssertionError('unexpected scoring')):
            self.assertEqual(sample_rows(logits,[left,right]),[0,1])
        right.logprobs_top=1
        self.assertEqual(sample_rows(logits,[left,right]),[0,1])
        self.assertIsNone(left.last_logprobs)
        self.assertEqual(right.last_logprobs['token_id'],1)

    def test_invalid_options_and_nonfinite_scores(self):
        for enabled,top in ((1,0),(True,True),(True,-1),(True,21),(False,1),(True,1.5)):
            with self.subTest(enabled=enabled,top=top),self.assertRaises(ValueError):options(enabled,top)
        self.assertIsNone(options(None,None))
        self.assertEqual(options(True,0),0)
        with self.assertRaises(ValueError):scores(torch.tensor([float('nan'),0.]),0,1)

    def test_batch_keeps_different_top_counts_and_unscored_rows_isolated(self):
        samplers=[Sampler(Sampling()) for _ in range(3)]
        samplers[0].logprobs_top=0;samplers[2].logprobs_top=2
        logits=torch.tensor([[3.,1.,0.],[1.,4.,2.],[0.,1.,3.]])
        self.assertEqual(sample_rows(logits,samplers),[0,1,2])
        self.assertEqual(samplers[0].last_logprobs['top'],[])
        self.assertIsNone(samplers[1].last_logprobs)
        self.assertEqual([token for token,_ in samplers[2].last_logprobs['top']],[2,1])
        for row in (0,2):
            packet=samplers[row].last_logprobs
            self.assertAlmostEqual(packet['logprob'],torch.log_softmax(logits[row],0)[packet['token_id']].item())

    def test_bytelevel_unicode_fragments_and_lifetime(self):
        tokenizer=Tokenizer(models.BPE({'ð':0,'Ł':1,'¦':2,'Ļ':3,'Ġ':4,'hello':5},[]))
        tokenizer.decoder=decoders.ByteLevel()
        text=TokenText(tokenizer)
        raw=b''.join(bytes(text.token(i)[1]) for i in range(4))
        self.assertEqual(raw.decode(),'🦙')
        self.assertEqual(text.token(4),(' ',(32,)))
        ref=weakref.ref(text);del text;gc.collect()
        self.assertIsNone(ref())

    def test_byte_fallback_and_unknown_decoder_bytes(self):
        tokenizer=Tokenizer(models.WordLevel({'<0xE2>':0,'<0x82>':1,'<0xAC>':2}))
        tokenizer.decoder=decoders.ByteFallback()
        text=TokenText(tokenizer)
        self.assertEqual(b''.join(bytes(text.token(i)[1]) for i in range(3)).decode(),'€')
        tokenizer.decoder=decoders.WordPiece()
        self.assertIsNone(TokenText(tokenizer).token(0)[1])

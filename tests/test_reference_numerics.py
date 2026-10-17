from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from damendaemon.floating_attention import _norm
from damendaemon.hyper_connections import reduce_head, reduce_streams, restore_streams, mixing_weights
from damendaemon.deepseek_generation import ReferenceGenerator
from tests.test_deepseek_generation import text_fixture, ScriptedModel
from tests.test_floating_model import fixture


class ReferenceNumericsTests(unittest.TestCase):
    def test_finite_variance_overflow_cannot_turn_into_zero(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.assertRaisesRegex(ValueError, 'variance'):
                _norm(torch.full((1,1,2),1e20,dtype=dtype),torch.ones(2),1e-6)
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaisesRegex(ValueError,'normalization input'):
                _norm(torch.tensor([value]),torch.ones(1),1e-6)
            with self.assertRaisesRegex(ValueError,'normalization weight'):
                _norm(torch.ones(1),torch.tensor([value]),1e-6)
        with self.assertRaisesRegex(ValueError,'normalization output'):
            _norm(torch.tensor([1.,0.],dtype=torch.float16),torch.full((2,),65504.),1e-6)

    def test_projection_overflow_cannot_be_hidden_by_sigmoid(self):
        hidden=torch.ones(1,1,2,2);maximum=torch.finfo(torch.float32).max
        with self.assertRaisesRegex(ValueError,'head projection'):
            reduce_head(hidden,torch.full((2,4),maximum),torch.ones(1),torch.zeros(2))
        with self.assertRaisesRegex(ValueError,'hyper-connection projection'):
            reduce_streams(hidden,torch.full((8,4),maximum),torch.ones(3),torch.zeros(8))
        for scale in (torch.tensor([maximum]), torch.tensor([float('inf')])):
            with self.assertRaises(ValueError):
                reduce_head(hidden,torch.ones(2,4),scale,torch.zeros(2))
        with self.assertRaisesRegex(ValueError,'variance'):
            reduce_head(hidden*1e20,torch.zeros(2,4),torch.ones(1),torch.zeros(2))

    def test_mixing_and_restore_reject_overflow_before_normalization_or_cast(self):
        maximum=torch.finfo(torch.float32).max
        for index in (0,1,2):
            scale=torch.ones(3);scale[index]=maximum
            with self.assertRaises(ValueError):
                mixing_weights(torch.full((1,1,8),2.),scale,torch.zeros(8),2)
        with self.assertRaisesRegex(ValueError,'column sum'):
            mixing_weights(torch.zeros(1,1,8),torch.ones(3),torch.zeros(8),2,epsilon=maximum)
        with self.assertRaisesRegex(ValueError,'restored streams'):
            restore_streams(torch.full((1,1,2),65504.,dtype=torch.float16),
                            torch.full((1,1,2,2),65504.,dtype=torch.float16),
                            torch.ones(1,1,2),torch.ones(1,1,2,2))

    def test_head_failure_releases_prefill_and_poisons_advanced_decode(self):
        model,ids=fixture();owned=[]
        with ExitStack() as stack:
            for block in model.blocks:
                original=block.prefill
                def record(*args, _original=original, **kwargs):
                    result=_original(*args,**kwargs);owned.append(result[1]);return result
                stack.enter_context(patch.object(block,'prefill',side_effect=record))
            linear=torch.nn.functional.linear
            def overflow_head(value, weight, *args, **kwargs):
                if weight is model.head['head.weight']:
                    return torch.full((1,4),float('inf'))
                return linear(value,weight,*args,**kwargs)
            with patch('damendaemon.floating_model.F.linear',side_effect=overflow_head):
                with self.assertRaisesRegex(ValueError,'vocabulary logits'):model.prefill(ids[:,:3])
        self.assertTrue(owned);self.assertTrue(all(state.attention.closed for state in owned))
        self.assertFalse(model.failed_prefills)
        _,state=model.prefill(ids[:,:3],max_tokens=9)
        # Mutating caller-owned prepared weights is unsupported, but still must
        # not silently turn invalid arithmetic into a successful generation.
        model.head['head.weight'].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'vocabulary weight'):
            model.decode(ids[:,3:4],state,position=3)
        self.assertTrue(state.poisoned);self.assertEqual(state.position,3)
        self.assertTrue(all(layer.attention.poisoned for layer in state.layers))
        state.close()

    def test_reference_attention_indexer_failure_poisons_decode_and_rejects_prefill(self):
        from tests.test_floating_attention import fixture as attention_fixture
        from damendaemon.floating_attention import attention_prefill,attention_decode
        hidden,weights,freq=attention_fixture(4,5)
        _,state=attention_prefill(hidden[:,:4],weights,freq,groups=2,window=4,
                                  ratio=4,index_top_k=2,max_tokens=6)
        self.addCleanup(state.close)
        original=torch.nn.functional.linear
        def overflow_query(value,weight,*args,**kwargs):
            if weight is weights['indexer.wq_b.weight']:
                return torch.full((*value.shape[:-1],weight.shape[0]),float('inf'))
            return original(value,weight,*args,**kwargs)
        with patch('damendaemon.floating_indexer.linear',side_effect=overflow_query):
            with self.assertRaisesRegex(ValueError,'indexer intermediate'):
                attention_prefill(hidden,weights,freq,groups=2,window=4,ratio=4,index_top_k=2)
            with self.assertRaisesRegex(ValueError,'indexer intermediate'):
                attention_decode(hidden[:,4:5],state,position=4)
        self.assertTrue(state.poisoned)
        self.assertEqual(state.position,4)
        with self.assertRaises(RuntimeError):
            attention_decode(hidden[:,4:5],state,position=4)
        state.close();self.assertTrue(state.closed)

    def test_reference_compressor_failure_leaves_decode_state_reusable(self):
        from tests.test_floating_attention import fixture as attention_fixture
        from damendaemon.floating_attention import attention_prefill,attention_decode
        hidden,weights,freq=attention_fixture(4,4)
        _,state=attention_prefill(hidden[:,:3],weights,freq,groups=2,window=4,
                                  ratio=4,index_top_k=2,max_tokens=5)
        self.addCleanup(state.close)
        before=state.window.clone()
        original=torch.nn.functional.linear
        def oversized(value,weight,*args,**kwargs):
            if weight is weights['compressor.wkv.weight']:
                return torch.full((*value.shape[:-1],weight.shape[0]),1e20)
            return original(value,weight,*args,**kwargs)
        with patch('damendaemon.floating_compressor.F.linear',side_effect=oversized):
            with self.assertRaisesRegex(ValueError,'compressor intermediate'):
                attention_prefill(hidden,weights,freq,groups=2,window=4,ratio=4,index_top_k=2)
            with self.assertRaisesRegex(ValueError,'compressor intermediate'):
                attention_decode(hidden[:,3:4],state,position=3)
        # Compression fails during pure preparation, before cache mutation.
        self.assertFalse(state.poisoned);self.assertEqual(state.position,3)
        self.assertEqual(state.pool.position,3)
        torch.testing.assert_close(state.window,before,equal_nan=True)
        actual=attention_decode(hidden[:,3:4],state,position=3)
        expected,fresh=attention_prefill(hidden,weights,freq,groups=2,window=4,ratio=4,index_top_k=2)
        try:torch.testing.assert_close(actual,expected[:,-1:],rtol=3e-5,atol=2e-6)
        finally:fresh.close()
        self.assertEqual(state.position,4)
        state.close();self.assertTrue(state.closed)

    def test_generation_checks_all_logits_before_sampling_and_closes_state(self):
        with tempfile.TemporaryDirectory() as temp:
            text=text_fixture(Path(temp));model=ScriptedModel(text,'abcdef')
            runner=ReferenceGenerator(model,text,max_context=128);messages=[dict(role='user',content='Hi')]
            for logits in (torch.full((1,text.vocab_size),float('nan')),
                           torch.full((1,text.vocab_size),float('inf')),
                           torch.zeros(text.vocab_size),torch.zeros(1,text.vocab_size,dtype=torch.float16)):
                with patch.object(model,'logits',return_value=logits):
                    with self.assertRaises(RuntimeError):list(runner.generate(messages,max_new_tokens=2))
                self.assertTrue(model.states[-1].closed)
            generator=runner.generate(messages,max_new_tokens=4);next(generator)
            with patch.object(model,'decode',return_value=torch.full((1,text.vocab_size),float('nan'))):
                with self.assertRaisesRegex(RuntimeError,'nonfinite'):list(generator)
            self.assertTrue(model.states[-1].closed)

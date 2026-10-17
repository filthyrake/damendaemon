"""CPU source failures and metadata-only hybrid capacity planning."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import torch
from damendaemon.config import hybrid_config
from damendaemon.hybrid import load_weights
from tests.test_capacity_cli import fixture


class HybridCheckpointContractTests(unittest.TestCase):
    def test_prefixes_tied_heads_and_malformed_inventory(self):
        with TemporaryDirectory() as folder:
            raw,original=fixture(Path(folder));c=hybrid_config(raw)
        for wrapped in (True,False):
            with self.subTest(wrapped=wrapped):
                state={name if wrapped else name.replace('model.language_model.','model.'):value for name,value in original.items()}
                prefix='model.language_model.' if wrapped else 'model.'
                expected=load_weights(dict(state),c)
                with self.assertRaisesRegex(ValueError,'parameter dtype'):
                    load_weights({**state,prefix+'norm.weight':state[prefix+'norm.weight'].to(torch.int8)},c)
                state['lm_head.weight']=state[prefix+'embed_tokens.weight'].clone()
                actual=load_weights(dict(state),c)
                for name in expected:torch.testing.assert_close(actual[name],expected[name])
                self.assertIs(actual['embed'],actual['head'])
                for name,value in state.items():
                    with self.subTest(shape=name),self.assertRaisesRegex(ValueError,'shape mismatch'):
                        load_weights({**state,name:value.unsqueeze(0)},c)
                    if name!='lm_head.weight':
                        missing=dict(state);del missing[name]
                        with self.subTest(missing=name),self.assertRaises(ValueError):load_weights(missing,c)
                with self.assertRaisesRegex(ValueError,'unequal tied'):
                    load_weights({**state,'lm_head.weight':state['lm_head.weight']+1},c)
                with self.assertRaisesRegex(ValueError,'unexpected'):
                    load_weights({**state,'unused.weight':torch.ones(1)},c)
                with self.assertRaises(ValueError):
                    load_weights({**state,('model.' if wrapped else 'model.language_model.')+'embed_tokens.weight':state['lm_head.weight']},c)
                extra={**state,'mtp.ignored':torch.ones(1),'model.visual.ignored':torch.ones(1)}
                self.assertIsNotNone(load_weights(extra,c))
                self.assertEqual(set(extra),{'mtp.ignored','model.visual.ignored'})
                untied={**state,'lm_head.weight':state['lm_head.weight']+1}
                result=load_weights(untied,{**c,'tie_word_embeddings':False})
                torch.testing.assert_close(result['head'],state['lm_head.weight'].half()+1)
                meta={name:torch.empty_like(value,device='meta') for name,value in state.items()}
                with patch('damendaemon.device.selected_accelerator',side_effect=AssertionError('hardware queried')),patch('damendaemon.layer_stream.enabled',return_value=False):
                    planned=load_weights(meta,c)
                self.assertEqual(meta,{})
                self.assertEqual(set(planned),set(expected))

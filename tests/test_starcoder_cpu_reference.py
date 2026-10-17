"""StarCoder2 checkpoint contracts and independent CPU numerical regression."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import torch
from transformers import Starcoder2Config,Starcoder2ForCausalLM
from tokenizers import Tokenizer,models
from damendaemon.qwen2 import ModelWeights
from damendaemon.starcoder import StarcoderExecutor,load_weights
from damendaemon.config import decoder_config


class StarcoderCPUReferenceTests(unittest.TestCase):
    def test_checkpoint_contract_and_logits(self):
        for bias in (True,False):
            with self.subTest(bias=bias),TemporaryDirectory() as folder,torch.random.fork_rng(devices=[]):
                torch.manual_seed(73)
                root=Path(folder)
                c=Starcoder2Config(vocab_size=32,hidden_size=8,intermediate_size=16,
                    num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                    max_position_embeddings=32,sliding_window=4,use_bias=bias,eos_token_id=1)
                c._attn_implementation='eager'
                reference=Starcoder2ForCausalLM(c).eval();reference.save_pretrained(root)
                Tokenizer(models.WordLevel({'unknown':0,'eos':1},unk_token='unknown')).save(str(root/'tokenizer.json'))
                (root/'tokenizer_config.json').write_text('{}')
                with patch('damendaemon.qwen2.selected_device',return_value='cpu'),patch('damendaemon.qwen2.empty_cache'),patch('damendaemon.qwen2.available_memory',return_value=2**30):
                    loaded=ModelWeights(root,16)
                    executor=StarcoderExecutor(loaded,use_graph=False,kernels='torch',max_batch=2)
                prompt=[3,4,5,6,7,8]
                actual=executor.prefill(prompt,slot=1)
                with torch.inference_mode():expected=reference(torch.tensor([prompt])).logits[:,-1]
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                actual,_=executor.decode_batch([9],[1])
                with torch.inference_mode():expected=reference(torch.tensor([prompt+[9]])).logits[:,-1]
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                parsed=decoder_config(c.to_dict())
                state=dict(reference.state_dict())
                weights=load_weights(dict(state),parsed)
                with self.assertRaisesRegex(ValueError,'parameter dtype'):
                    load_weights({**state,'model.norm.weight':state['model.norm.weight'].to(torch.int8)},parsed)
                self.assertIs(weights['embed'],weights['head'])
                corrupt=dict(state);corrupt['lm_head.weight']=state['lm_head.weight']+1
                with self.assertRaisesRegex(ValueError,'unequal tied'):load_weights(corrupt,parsed)
                corrupt=dict(state);corrupt['unused.weight']=torch.ones(1)
                with self.assertRaisesRegex(ValueError,'unexpected'):load_weights(corrupt,parsed)
                for name,value in state.items():
                    corrupt=dict(state);corrupt[name]=value.unsqueeze(0)
                    with self.subTest(tensor=name),self.assertRaisesRegex(ValueError,'shape mismatch'):
                        load_weights(corrupt,parsed)
                    if name!='lm_head.weight':
                        missing=dict(state);del missing[name]
                        with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'missing'):
                            load_weights(missing,parsed)
                untied=dict(state);untied['lm_head.weight']=state['lm_head.weight']+1
                weights=load_weights(untied,{**parsed,'tie_word_embeddings':False})
                torch.testing.assert_close(weights['head'],state['lm_head.weight']+1)

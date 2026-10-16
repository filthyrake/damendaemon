"""Independent checkpoint layouts, with CPU-only source-contract failures."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import torch
from transformers import (Qwen2Config,Qwen2ForCausalLM,Qwen3Config,Qwen3ForCausalLM,
    LlamaConfig,LlamaForCausalLM,MistralConfig,MistralForCausalLM,Phi3Config,Phi3ForCausalLM)
from tokenizers import Tokenizer,models
from safetensors.torch import save_file
from dllm.config import decoder_config
from dllm.dense_checkpoint import validate
from dllm.qwen2 import ModelWeights


class DenseCheckpointTests(unittest.TestCase):
    def test_supported_families_and_malformed_source_tensors(self):
        for config_class,model_class in ((Qwen2Config,Qwen2ForCausalLM),(Qwen3Config,Qwen3ForCausalLM),
                (LlamaConfig,LlamaForCausalLM),(MistralConfig,MistralForCausalLM),(Phi3Config,Phi3ForCausalLM)):
            with self.subTest(family=config_class.model_type),TemporaryDirectory() as folder,torch.random.fork_rng(devices=[]):
                torch.manual_seed(73)
                root=Path(folder)
                options=dict(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=32,
                    eos_token_id=1,tie_word_embeddings=True)
                if config_class is Phi3Config:options.update(original_max_position_embeddings=32,pad_token_id=0)
                c=config_class(**options)
                reference=model_class(c).eval();reference.save_pretrained(root)
                Tokenizer(models.WordLevel({'unknown':0,'eos':1},unk_token='unknown')).save(str(root/'tokenizer.json'))
                (root/'tokenizer_config.json').write_text('{}')
                parsed=decoder_config(c.to_dict());state=dict(reference.state_dict())
                validate(state,parsed)
                with patch('dllm.qwen2.selected_device',return_value='cpu'):
                    loaded=ModelWeights(root,16)
                self.assertIs(loaded.weights['head'],loaded.weights['embed'])
                for name,value in state.items():
                    corrupt={**state,name:value.unsqueeze(0)}
                    with self.subTest(shape=name),self.assertRaisesRegex(ValueError,'shape mismatch'):validate(corrupt,parsed)
                    if name!='lm_head.weight':
                        missing=dict(state);del missing[name]
                        with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'missing'):validate(missing,parsed)
                with self.assertRaisesRegex(ValueError,'unexpected'):validate({**state,'unused.weight':torch.ones(1)},parsed)
                corrupt={**state,'lm_head.weight':state['lm_head.weight']+1}
                with self.assertRaisesRegex(ValueError,'unequal tied'):validate(corrupt,parsed)
                validate(corrupt,{**parsed,'tie_word_embeddings':False})
                # Exercise the real constructor boundary, not only the validator.
                save_file({name:value.clone() for name,value in corrupt.items()},str(root/'model.safetensors'))
                with patch('dllm.qwen2.selected_device',return_value='cpu'),patch.object(ModelWeights,'finalize_weights',side_effect=AssertionError('packing reached')):
                    with self.assertRaisesRegex(ValueError,'unequal tied'):ModelWeights(root,16)

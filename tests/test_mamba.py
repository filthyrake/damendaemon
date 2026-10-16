import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dllm.mamba import config,packable


class MambaTests(unittest.TestCase):
    def test_only_aligned_projections_are_quantized(self):
        self.assertTrue(packable('0.in_proj',(3072,768)))
        self.assertTrue(packable('backbone.layers.0.mixer.dt_proj.weight',(2048,64)))
        self.assertTrue(packable('head',(50280,768)))
        self.assertFalse(packable('0.dt_proj',(1536,48)))
        for name in ('embed','0.A','0.conv','0.norm'):
            self.assertFalse(packable(name,(1536,32)))

    def test_configuration_requires_qualified_recurrence_geometry(self):
        value={'model_type':'mamba','architectures':['MambaForCausalLM'],'hidden_size':768,
            'intermediate_size':1536,'state_size':16,'conv_kernel':4,'time_step_rank':48,
            'num_hidden_layers':24,'vocab_size':50280,'eos_token_id':0}
        with TemporaryDirectory() as directory:
            file=Path(directory)/'config.json';file.write_text(json.dumps(value))
            parsed=config(directory)
            self.assertEqual(parsed['max_position_embeddings'],2048)
            self.assertTrue(parsed['tie_word_embeddings'])
            for change in ({'layer_norm_epsilon':10**400},{'state_size':33},{'conv_kernel':17},{'time_step_rank':0},{'hidden_act':'gelu'},
                           {'rms_norm':False},{'use_bias':1},{'layer_norm_epsilon':0},{'eos_token_id':50280},
                           {'architectures':['Mamba2ForCausalLM']},{'quantization_config':{'bits':4}}):
                file.write_text(json.dumps({**value,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):config(directory)

    def test_saved_eos_replaces_default_before_weights(self):
        from transformers import MambaConfig,MambaForCausalLM,GenerationConfig
        from tokenizers import Tokenizer,models
        from dllm.mamba import MambaWeights
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            reference=MambaForCausalLM(MambaConfig(hidden_size=8,state_size=4,
                num_hidden_layers=1,expand=2,conv_kernel=4,time_step_rank=2,
                vocab_size=16,eos_token_id=1))
            reference.save_pretrained(directory)
            Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(Path(directory)/'tokenizer.json'))
            generation=Path(directory)/'generation_config.json'
            generation.write_text(json.dumps({'eos_token_id':[5,6]}))
            expected=GenerationConfig.from_pretrained(directory,local_files_only=True)
            self.assertEqual(MambaWeights(directory,16).eos,set(expected.eos_token_id))
            generation.write_text('{}')
            self.assertEqual(MambaWeights(directory,16).eos,{1})
            for bad in (None,True,[],[16],-1,'5'):
                generation.write_text(json.dumps({'eos_token_id':bad}))
                with self.subTest(bad=bad),patch('dllm.qwen2.weight_files',side_effect=AssertionError('weights opened')):
                    with self.assertRaises(ValueError):MambaWeights(directory,16)
            generation.write_text('{"eos_token_id":5,"eos_token_id":6}')
            with self.assertRaises(ValueError):config(directory)

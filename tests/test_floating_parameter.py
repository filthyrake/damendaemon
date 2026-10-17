"""Reject undeclared packed payloads using real CPU checkpoint files."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from safetensors.torch import load_file,save_file
from tokenizers import Tokenizer,models
from damendaemon.floating_parameter import require_floating


class FloatingParameterTests(unittest.TestCase):
    def test_supported_and_unsupported_dtypes_without_conversion(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            value=torch.ones(2,dtype=dtype)
            self.assertIs(require_floating(value,'weight'),value)
        for dtype in (torch.int8,torch.uint8,torch.int32,torch.int64,torch.bool,torch.complex64,torch.float8_e4m3fn):
            with self.subTest(dtype=dtype),self.assertRaisesRegex(ValueError,'parameter dtype'):
                require_floating(torch.empty(2,dtype=dtype),'weight')

    def test_real_encoder_and_recurrent_snapshots_reject_integer_parameters(self):
        from transformers import (T5Config,T5ForConditionalGeneration,MambaConfig,MambaForCausalLM,
            Mamba2Config,Mamba2ForCausalLM,BertConfig,BertForSequenceClassification)
        from damendaemon.t5 import T5Weights
        from damendaemon.mamba import MambaWeights
        from damendaemon.mamba2 import Mamba2Weights
        from damendaemon.encoder import EncoderWeights
        cases=(
            (T5ForConditionalGeneration,T5Config(d_model=8,d_ff=16,d_kv=4,num_heads=2,num_layers=1,
                vocab_size=16,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1),T5Weights),
            (MambaForCausalLM,MambaConfig(hidden_size=8,state_size=4,num_hidden_layers=1,expand=2,
                conv_kernel=4,time_step_rank=2,vocab_size=16,eos_token_id=1),MambaWeights),
            (Mamba2ForCausalLM,Mamba2Config(hidden_size=8,state_size=4,num_hidden_layers=1,expand=2,
                conv_kernel=4,num_heads=4,head_dim=4,n_groups=1,vocab_size=16,eos_token_id=1),Mamba2Weights),
            (BertForSequenceClassification,BertConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                num_attention_heads=2,vocab_size=16,max_position_embeddings=32,num_labels=1,
                problem_type='regression'),EncoderWeights))
        for model_class,config,loader in cases:
            with self.subTest(family=config.model_type),TemporaryDirectory() as folder:
                root=Path(folder);model_class(config).save_pretrained(root)
                Tokenizer(models.WordLevel({'unknown':0},unk_token='unknown')).save(str(root/'tokenizer.json'))
                loader(root,8)
                file=root/'model.safetensors';state=load_file(str(file));name=next(iter(state))
                state[name]=state[name].to(torch.int8);save_file(state,str(file))
                with self.assertRaisesRegex(ValueError,'parameter dtype'):loader(root,8)

    def test_legacy_decoder_importers_reject_integer_parameters(self):
        from transformers import (GPT2Config,GPT2LMHeadModel,BloomConfig,BloomForCausalLM,
            GPTNeoXConfig,GPTNeoXForCausalLM)
        from damendaemon.config import decoder_config
        from damendaemon.gpt2 import load_weights as gpt2_weights
        from damendaemon.bloom import load_weights as bloom_weights
        from damendaemon.neox import load_weights as neox_weights
        cases=((GPT2LMHeadModel,GPT2Config(n_embd=8,n_head=2,n_layer=1,n_positions=32,vocab_size=16),gpt2_weights),
               (BloomForCausalLM,BloomConfig(hidden_size=8,n_head=2,n_layer=1,vocab_size=16),bloom_weights),
               (GPTNeoXForCausalLM,GPTNeoXConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                   num_attention_heads=2,rotary_pct=.5,max_position_embeddings=32,vocab_size=16),neox_weights))
        for model_class,config,loader in cases:
            with self.subTest(family=config.model_type):
                config.architectures=[model_class.__name__]
                state=dict(model_class(config).state_dict());parsed=decoder_config(config.to_dict())
                loader(dict(state),parsed)
                name=next(iter(state));state[name]=state[name].to(torch.int8)
                with self.assertRaisesRegex(ValueError,'parameter dtype'):loader(state,parsed)

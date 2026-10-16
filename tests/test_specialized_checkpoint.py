"""Gemma/OLMoE source contracts against independent CPU model layouts."""
import unittest
import torch
from transformers import Gemma3TextConfig,Gemma3ForCausalLM,OlmoeConfig,OlmoeForCausalLM
from dllm.config import decoder_config
from dllm.gemma import load_weights as gemma_weights
from dllm.moe import load_weights as olmoe_weights


class SpecializedCheckpointTests(unittest.TestCase):
    def test_source_contracts_and_transforms(self):
        for config_class,model_class,loader in ((Gemma3TextConfig,Gemma3ForCausalLM,gemma_weights),
                                               (OlmoeConfig,OlmoeForCausalLM,olmoe_weights)):
            for tied in (False,True):
                with self.subTest(family=config_class.model_type,tied=tied),torch.random.fork_rng(devices=[]):
                    torch.manual_seed(73)
                    options=dict(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                        num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=32,
                        eos_token_id=1,pad_token_id=0,tie_word_embeddings=tied)
                    if config_class is Gemma3TextConfig:
                        options.update(layer_types=['sliding_attention'],sliding_window=4,query_pre_attn_scalar=4)
                    else:options.update(num_experts=4,num_experts_per_tok=2)
                    c=config_class(**options);reference=model_class(c)
                    parsed=decoder_config(c.to_dict());state=dict(reference.state_dict())
                    weights=loader(dict(state),parsed)
                    with self.assertRaisesRegex(ValueError,'parameter dtype'):
                        loader({**state,'model.norm.weight':state['model.norm.weight'].to(torch.int8)},parsed)
                    dtype=torch.bfloat16 if loader is gemma_weights else torch.float16
                    torch.testing.assert_close(weights['embed'],state['model.embed_tokens.weight'].to(dtype))
                    if tied:self.assertIs(weights['head'],weights['embed'])
                    torch.testing.assert_close(weights['norm'],state['model.norm.weight'].float()+(1 if loader is gemma_weights else 0))
                    for name,value in state.items():
                        corrupt={**state,name:value.unsqueeze(0)}
                        with self.subTest(shape=name),self.assertRaisesRegex(ValueError,'shape mismatch'):loader(corrupt,parsed)
                        if name!='lm_head.weight' or not tied:
                            missing=dict(state);del missing[name]
                            with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'missing'):loader(missing,parsed)
                    with self.assertRaisesRegex(ValueError,'unexpected'):loader({**state,'unused.weight':torch.ones(1)},parsed)
                    corrupt={**state,'lm_head.weight':state['lm_head.weight']+1}
                    if tied:
                        with self.assertRaisesRegex(ValueError,'unequal tied'):loader(corrupt,parsed)
                    else:
                        expected=corrupt['lm_head.weight'].to(dtype)
                        torch.testing.assert_close(loader(corrupt,parsed)['head'],expected)

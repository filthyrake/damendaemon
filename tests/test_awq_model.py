import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from dllm.awq_model import AwqWeights, config, dense_groups, plan, projection_groups, weight_bytes


def fixture(path):
    c = dict(model_type='qwen2', architectures=['Qwen2ForCausalLM'], hidden_size=128,
             intermediate_size=256, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
             head_dim=64, vocab_size=32, max_position_embeddings=128, rms_norm_eps=1e-6,
             rope_theta=10000., hidden_act='silu', tie_word_embeddings=True, eos_token_id=1,
             quantization_config=dict(bits=4, group_size=128, version='gemm', quant_method='awq', zero_point=True))
    (path/'config.json').write_text(json.dumps(c))
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '[EOS]': 1, 'hello': 2, 'world': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path/'tokenizer.json'))
    (path/'tokenizer_config.json').write_text(json.dumps(dict(eos_token='[EOS]', chat_template="{{ messages[0]['content'] }}")))
    values = {name: torch.ones(shape, dtype=torch.float16) for _, entries in dense_groups(c) for name, shape in entries}
    values['lm_head.weight'] = values['model.embed_tokens.weight'].clone()
    for i, (_, entries) in enumerate(projection_groups(c)):
        for j, (prefix, inputs, outputs) in enumerate(entries):
            values[prefix+'.qweight'] = torch.full((inputs, outputs//8), 0x76543210, dtype=torch.int32)
            values[prefix+'.qzeros'] = torch.full((inputs//128, outputs//8), 0x88888888 - 2**32, dtype=torch.int32)
            values[prefix+'.scales'] = torch.full((inputs//128, outputs), .01*(i+j+1), dtype=torch.float16)
    save_file(values, str(path/'model.safetensors'))
    return c, values


class AwqModelTests(unittest.TestCase):
    def test_qwen3_qualification_requires_norms_without_qkv_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            c, values = fixture(path)
            c.update(model_type='qwen3', architectures=['Qwen3ForCausalLM'], attention_bias=False)
            (path/'config.json').write_text(json.dumps(c))
            for name in list(values):
                if name.endswith('_proj.bias'):
                    del values[name]
            for kind in ('q', 'k'):
                values[f'model.layers.0.self_attn.{kind}_norm.weight'] = torch.ones(64, dtype=torch.float16)
            values = {name: value.to(torch.bfloat16) if name.endswith('.weight') else value
                      for name, value in values.items()}
            save_file(values, str(path/'model.safetensors'))
            from dllm.models import load
            model = load(path, 64, 'none')
            self.assertNotIn('0.qkv.bias', model.weights)
            self.assertEqual(model.weights['embed'].dtype, torch.bfloat16)
            self.assertEqual(model.bytes, plan(path).host_bytes)
            self.assertEqual(tuple(model.weights['0.q_norm'].shape), (64,))
            self.assertEqual(tuple(model.weights['0.k_norm'].shape), (64,))
            values['model.embed_tokens.weight'][0, 0] = 1e10
            save_file(values, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'FP16 execution range'):
                AwqWeights(path, 64, 'none')
            values['model.embed_tokens.weight'][0, 0] = 1
            del values['model.layers.0.self_attn.k_norm.weight']
            save_file(values, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'missing AWQ tensors'):
                AwqWeights(path, 64, 'none')

    def test_shared_factory_preserves_packed_checkpoint_and_rejects_other_formats(self):
        from dllm import models as factory
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            c, _ = fixture(path)
            self.assertEqual(factory.config(path)['weight_format'], 'awq_gemm')
            model = factory.load(path, 64, 'none')
            self.assertIsInstance(model, AwqWeights)
            self.assertEqual(factory.weight_bytes(path, 'none'), model.bytes)
            with self.assertRaisesRegex(ValueError, 'checkpoint-native'):
                factory.load(path, 64, 'q8_0')
            for change in ({'version': 'gemv'}, {'bits': 8}, {'quant_method': 'gptq'}):
                altered = {**c, 'quantization_config': {**c['quantization_config'], **change}}
                (path/'config.json').write_text(json.dumps(altered))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    factory.config(path)

    def test_packed_fusion_tied_head_and_admission_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            model = AwqWeights(path, 64, 'none')
            self.assertEqual(model.bytes, weight_bytes(path, 'none'))
            self.assertEqual(model.weight_format, 'awq_gemm')
            self.assertIs(model.weights['head'], model.weights['embed'])
            self.assertEqual(model.weights['0.qkv.weight'].dtype, torch.int32)
            self.assertEqual(tuple(model.weights['0.qkv.weight'].shape), (128, 32))
            self.assertEqual(tuple(model.awq_matrices['0.gate_up'].zeros.shape), (1, 64))
            self.assertEqual(model.chat_tokens([{'role': 'user', 'content': 'hello world'}]), [2, 3])
            self.assertEqual(len(model.awq_matrices), 4)

    def test_bad_headers_and_missing_or_extra_modules_fail_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, original = fixture(path)
            key = 'model.layers.0.self_attn.q_proj.scales'
            for change in ('dtype', 'shape', 'missing', 'extra'):
                values = original.copy()
                if change == 'dtype': values[key] = values[key].float()
                elif change == 'shape': values[key] = values[key][:, :8].contiguous()
                elif change == 'missing': values.pop(key)
                else: values['model.unexpected.weight'] = torch.ones(1)
                save_file(values, str(path/'model.safetensors'))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    plan(path)

    def test_shard_index_escape_mismatch_and_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, values = fixture(path)
            prepared = plan(path)
            index = path/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'x': '../escape.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'invalid checkpoint shard path'):
                plan(path)
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepared.check_source()
            index.write_text(json.dumps({'weight_map': {next(iter(values)): 'model.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'shard index'):
                plan(path)

    def test_index_growth_after_discovery_keeps_second_read_bounded(self):
        from dllm import awq_model
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);_,values=fixture(path)
            index=path/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{name:'model.safetensors' for name in values}}))
            discover=awq_model.weight_files
            def grow_after_discovery(root):
                files=discover(root)
                with index.open('a') as writer:writer.write(' '*(16*1024*1024))
                return files
            with patch.object(awq_model,'weight_files',side_effect=grow_after_discovery):
                with self.assertRaisesRegex(ValueError,'16 MiB metadata limit'):plan(path)

    def test_tied_head_disagreement_and_nonfinite_scales_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, values = fixture(path)
            values['lm_head.weight'] = values['lm_head.weight'] + 1
            save_file(values, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'tied head'):
                AwqWeights(path, 64, 'none')
            values['lm_head.weight'] = values['model.embed_tokens.weight'].clone()
            values['model.layers.0.self_attn.q_proj.scales'][0, 0] = float('nan')
            save_file(values, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'finite'):
                AwqWeights(path, 64, 'none')

    def test_unsupported_device_conversion_and_architecture_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            c, _ = fixture(path)
            with self.assertRaisesRegex(ValueError, 'requantized'):
                weight_bytes(path, 'q4_0')
            with patch.dict('os.environ', {'DLLM_DEVICE': 'mps'}), self.assertRaisesRegex(ValueError, 'CUDA'):
                weight_bytes(path, 'none')
            c['model_type'] = 'mistral'
            c['architectures'] = ['MistralForCausalLM']
            (path/'config.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'Qwen2'):
                config(path)

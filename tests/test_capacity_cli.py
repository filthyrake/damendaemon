import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from dllm.capacity import plan_local
from dllm.cli import main
from tests import test_config


def fixture(path):
    config = test_config.HybridConfigTests().config()
    config['text_config'].update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=64,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=16,
        linear_value_head_dim=16, layer_types=['linear_attention', 'full_attention'])
    shapes = {'embed_tokens.weight': (64, 32), 'norm.weight': (32,)}
    for i in range(2):
        shapes.update({f'layers.{i}.'+name: shape for name, shape in {
            'input_layernorm.weight': (32,), 'post_attention_layernorm.weight': (32,),
            'mlp.gate_proj.weight': (64, 32), 'mlp.up_proj.weight': (64, 32),
            'mlp.down_proj.weight': (32, 64)}.items()})
    shapes.update({'layers.0.linear_attn.'+name: shape for name, shape in {
        'in_proj_qkv.weight': (64, 32), 'in_proj_z.weight': (32, 32),
        'in_proj_b.weight': (2, 32), 'in_proj_a.weight': (2, 32),
        'conv1d.weight': (64, 1, 4), 'out_proj.weight': (32, 32),
        'dt_bias': (2,), 'A_log': (2,), 'norm.weight': (16,)}.items()})
    shapes.update({'layers.1.self_attn.'+name: shape for name, shape in {
        'q_proj.weight': (64, 32), 'k_proj.weight': (16, 32), 'v_proj.weight': (16, 32),
        'o_proj.weight': (32, 32), 'q_norm.weight': (16,), 'k_norm.weight': (16,)}.items()})
    tensors = {'model.language_model.'+name: torch.zeros(shape) for name, shape in shapes.items()}
    (path/'config.json').write_text(json.dumps(config))
    save_file(tensors, str(path/'model.safetensors'))
    return config, tensors


class CapacityCliTests(unittest.TestCase):
    def test_header_plan_matches_runtime_weight_budget_including_tied_packed_head(self):
        from dllm.qwen2 import weight_bytes
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            with patch.dict('os.environ', DLLM_DEVICE='rocm', DLLM_EXPERIMENTAL_VISION='0'), \
                    patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU initialized')), \
                    patch('dllm.rocm_policy.require_host_headroom', side_effect=AssertionError('meta mapping read host reserve')):
                report = plan_local(path, [512, 32768], [1, 4], 'float32')
                with patch('dllm.hybrid_q4_policy.validate'):
                    for mode in ('none', 'q8_0', 'q4_0'):
                        self.assertEqual(report['weight_bytes'][mode], weight_bytes(path, mode))
            # One full-attention layer + one FP32 recurrent/convolution layer.
            state = 2*1*512*16*4 + 2*16*16*4 + 64*4*4
            self.assertEqual(report['contexts']['512']['1']['state_bytes'], state)
            self.assertEqual(report['contexts']['512']['4']['state_bytes'], state*4)
            streaming = report['layer_streaming']
            self.assertEqual(streaming['peak_resident_weight_bytes'],
                             streaming['fixed_weight_bytes'] + max(streaming['per_layer_weight_bytes']))
            self.assertEqual(streaming['output_head_workspace_bytes'], 64*32*4)
            self.assertEqual(streaming['contexts']['512']['1']['with_workspace_floor_bytes'],
                             streaming['peak_resident_weight_bytes'] + state + 64*32*4 + 512*1024**2)
            self.assertEqual(report['runtime_qualification'], 'not assessed')
            self.assertNotIn('runtime_supported_quantization', report)
            output = io.StringIO()
            with patch('sys.argv', ['dllm', 'plan-hybrid', directory, '--contexts', '512', '--batches', '1']), \
                    patch('sys.stdout', output), patch('dllm.cli.download', side_effect=AssertionError('downloaded')):
                main()
            self.assertEqual(json.loads(output.getvalue())['contexts']['512']['1']['state_bytes'], state)

    def test_invalid_header_geometry_index_dtype_and_limits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, tensors = fixture(path)
            for contexts, batches in (([0], [1]), ([262145], [1]), ([512], [0]), ([512], [17])):
                with self.subTest(contexts=contexts, batches=batches), self.assertRaises(ValueError):
                    plan_local(path, contexts, batches, 'float32')
            broken = dict(tensors)
            del broken['model.language_model.norm.weight']
            save_file(broken, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'weight mapping'):
                plan_local(path, [512], [1], 'float32')
            broken = dict(tensors)
            broken['model.language_model.norm.weight'] = torch.zeros(16)
            save_file(broken, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                plan_local(path, [512], [1], 'float32')
            broken['model.language_model.norm.weight'] = torch.zeros(32, dtype=torch.int8)
            save_file(broken, str(path/'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'floating-point'):
                plan_local(path, [512], [1], 'float32')
            save_file(tensors, str(path/'model.safetensors'))
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'wrong': 'model.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'shard index'):
                plan_local(path, [512], [1], 'float32')
            save_file(tensors, str(path/'duplicate.safetensors'))
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
                'first': 'model.safetensors', 'second': 'duplicate.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'duplicate tensor'):
                plan_local(path, [512], [1], 'float32')

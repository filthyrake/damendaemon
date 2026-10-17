import copy
from types import SimpleNamespace
import unittest
import torch
from damendaemon.stage_identity import fingerprint


class StageIdentityTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(config={'model_type': 'qwen2', 'hidden_size': 2}, max_context=512,
            quantization='q8_0', weights={'a': torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)},
            scales={'a': torch.tensor([.25, .5], dtype=torch.float16)})

    def test_identity_covers_configuration_weights_scales_and_policy(self):
        model = self.model(); expected = fingerprint(model)
        self.assertEqual(len(expected), 32)
        mutations = [lambda m: m.weights['a'].add_(1), lambda m: m.scales['a'].mul_(2),
                     lambda m: m.config.update(hidden_size=4), lambda m: setattr(m, 'max_context', 256),
                     lambda m: setattr(m, 'quantization', 'none'), lambda m: setattr(m, 'weight_format', 'gguf_q8_0')]
        for mutate in mutations:
            changed = copy.deepcopy(model); mutate(changed)
            self.assertNotEqual(fingerprint(changed), expected)
        self.assertEqual(fingerprint(model), expected)

    def test_logical_contents_ignore_cpu_strides_but_not_dtype(self):
        model = self.model(); expected = fingerprint(model)
        model.weights['a'] = model.weights['a'].t().contiguous().t()
        self.assertFalse(model.weights['a'].is_contiguous())
        self.assertEqual(fingerprint(model), expected)
        model.weights['a'] = model.weights['a'].to(torch.int16)
        self.assertNotEqual(fingerprint(model), expected)

    def test_non_cpu_storage_rejected_before_pointer_access(self):
        model = self.model(); model.weights['a'] = torch.empty((2, 2), device='meta')
        with self.assertRaisesRegex(ValueError, 'CPU tensors'):fingerprint(model)

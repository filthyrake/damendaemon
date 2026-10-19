"""Packed-weight hybrid pipeline admission and planning; CPU fixtures only."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch

from damendaemon.hybrid_layout import choose_stage_splits, stage_budget
from damendaemon.hybrid_pipeline_build import build_cuda_pipeline
from damendaemon.hybrid_pipeline_policy import validate as policy_validate
from damendaemon.hybrid_stage import shard_model
from tests import test_hybrid_stage
from tests.test_hybrid_pipeline_build import PipelineBuildTests


def pack_weights(model):
    """Q8_0-pack a fixture model's 2-D weights with per-32-channel FP16 scales."""
    from damendaemon.kv_quant import quantize_kv
    weights, scales = {}, {}
    for name, tensor in model.weights.items():
        if tensor.ndim == 2:
            packed, scale = quantize_kv(tensor.float())
            weights[name] = packed
            scales[name] = scale.reshape(tensor.shape[0], tensor.shape[1] // 32).half()
        else:
            weights[name] = tensor
    model.weights = weights
    model.scales = scales
    model.quantization = 'q8_0'
    return model


class PackedShardTests(unittest.TestCase):
    def setUp(self):
        fixture = test_hybrid_stage.HybridStageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.model = fixture.model

    def test_packed_shard_carries_stage_weights_and_scales(self):
        pack_weights(self.model)
        shard = shard_model(self.model, 0, 2)
        self.assertEqual(shard.quantization, 'q8_0')
        self.assertTrue(shard.scales)
        self.assertTrue(all(t.dtype == torch.int8 for t in shard.weights.values()
                            if t.ndim == 2))
        self.assertEqual(set(shard.scales), set(shard.weights) & set(self.model.scales))
        # Invalid ranges stay rejected for packed weights too.
        with self.assertRaises(ValueError):
            shard_model(self.model, 2, 2)
        with self.assertRaises(ValueError):
            shard_model(self.model, 1, 0)

    def test_packed_shard_requires_scales(self):
        pack_weights(self.model)
        self.model.scales = {}
        with self.assertRaisesRegex(ValueError, 'scales'):
            shard_model(self.model, 0, 2)

    def test_unquantized_shard_still_rejects_integer_weights(self):
        self.model.weights = {name: (torch.zeros(t.shape, dtype=torch.int16)
                                     if t.ndim == 2 else t)
                              for name, t in self.model.weights.items()}
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            shard_model(self.model, 0, 2)

    def test_stage_budget_counts_scale_bytes(self):
        pack_weights(self.model)
        plain = stage_budget(self.model.weights, self.model.config, 32, 2, 2, 0, 2)
        packed = stage_budget(self.model.weights, self.model.config, 32, 2, 2, 0, 2,
                              scales=self.model.scales)
        self.assertGreater(packed['scale_bytes'], 0)
        self.assertEqual(packed['required_memory_bytes'],
                         plain['required_memory_bytes'] + packed['scale_bytes'])
        placement = choose_stage_splits(self.model.weights, self.model.config, 32, 2, 2,
                                        [2**30, 2**30], scales=self.model.scales)
        self.assertTrue(placement['storage_plan_found'])


class QuantPipelinePolicyTests(unittest.TestCase):
    def setUp(self):
        fixture = test_hybrid_stage.HybridStageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.model = fixture.model
        self.config = self.model.config
        pack_weights(self.model)

    def policy(self, env, quantization='q8_0'):
        with patch.dict('os.environ', env, clear=True):
            policy_validate(self.config, quantization, 1024, 1, False)

    def test_quantized_pipeline_rejected_without_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'unquantized hybrid text weights'):
            self.policy({'DAMENDAEMON_DEVICE': 'cuda',
                         'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                         'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '2048,2048'})

    def test_quantized_pipeline_admitted_with_opt_in(self):
        self.policy({'DAMENDAEMON_DEVICE': 'cuda',
                     'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                     'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
                     'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '2048,2048'})

    def test_q4_pipeline_requires_compensated_precision(self):
        with self.assertRaisesRegex(ValueError, 'compensated'):
            self.policy({'DAMENDAEMON_DEVICE': 'cuda',
                         'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                         'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
                         'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '2048,2048'},
                        quantization='q4_0')
        with patch.dict('os.environ', {'DAMENDAEMON_DEVICE': 'cuda',
                                       'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                                       'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
                                       'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
                                       'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '2048,2048'}, clear=True):
            policy_validate(self.config, 'q4_0', 1024, 1, False)


class PackedPipelineBuildTests(PipelineBuildTests):
    """End-to-end CPU build of a packed two-stage pipeline (no CUDA execution).

    The packed-weights admission requires fused Triton kernels (the qualified
    CUDA packed path); the fixture stubs the kernels module since execution is
    not exercised on CPU.
    """

    def setUp(self):
        super().setUp()
        pack_weights(self.model)
        stub = patch.dict('sys.modules', {
            'damendaemon.hybrid_kernels': __import__('types').SimpleNamespace(),
            'damendaemon.kernels': __import__('types').SimpleNamespace()})
        stub.start()
        self.addCleanup(stub.stop)
        # The stubbed fused backend triggers a CUDA warmup probe at
        # construction; execution is not exercised on CPU.
        warmup = patch('damendaemon.cuda_warmup.warmup_stream', return_value=nullcontext())
        warmup.start()
        self.addCleanup(warmup.stop)
        forward = patch('damendaemon.hybrid.HybridExecutor.forward_batch',
                        return_value=torch.zeros(1, 64))
        forward.start()
        self.addCleanup(forward.stop)
        env = patch.dict('os.environ', {
            'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1'}, clear=False)
        env.start()
        self.addCleanup(env.stop)

    def build(self):
        return build_cuda_pipeline(self.model, [0, 1], [2**30, 2**30], max_batch=2)

    def test_packed_two_stage_build_succeeds_on_cpu_fixture(self):
        pipeline = self.build()
        try:
            self.assertEqual(pipeline.devices, (0, 1))
            self.assertEqual(len(pipeline.stages), 2)
            self.assertEqual(pipeline.stages[0].layer_start, 0)
            self.assertEqual(pipeline.stages[-1].layer_end, 4)
            self.assertTrue(all(stage.model.scales for stage in pipeline.stages))
            for stage, budget in zip(pipeline.stages, pipeline.placement['stages']):
                self.assertEqual(stage.required_memory_bytes, budget['required_memory_bytes'])
                self.assertLessEqual(stage.required_memory_bytes, 2**30)
        finally:
            pipeline.close()
        self.assertTrue(self.model.weights)

    def test_fitting_construction_executes_and_preserves_source(self):
        self.skipTest('packed CPU fixture stubs fused kernels; execution is not exercised here')


if __name__ == '__main__':
    unittest.main()

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

    def q4_identity(self, env, primary_device):
        """Run hybrid_q4_policy.validate with a faked CUDA device identity.

        Q4 admission is structural (kernels + device present, or the #536
        pipeline pair anchored on the A30 primary), so the only faked
        identity that matters is cuda:0's name for the pair anchor."""
        import damendaemon.hybrid_q4_policy as module
        with patch.dict('os.environ', env, clear=True), \
                patch.object(module, 'selected_accelerator', return_value='cuda'), \
                patch('torch.cuda.get_device_name', return_value=primary_device):
            module.validate(self.config, 512)

    def test_q4_stage_device_admitted_on_pipeline_pair(self):
        # #536 gate 2: stage 1 builds under the A2, so the pipeline pair
        # anchors identity on the A30 primary at cuda:0.
        self.q4_identity({'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
                          'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
                          'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                          'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
                          'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '22528,13312'},
                         primary_device='NVIDIA A30')

    def test_q4_stage_device_rejected_without_pipeline_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'Triton hybrid kernels'):
            self.q4_identity({'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
                              'DAMENDAEMON_HYBRID_PRECISION': 'compensated'},
                             primary_device='NVIDIA A2')

    def test_q4_stage_device_rejected_without_a30_primary(self):
        # The pipeline pair's identity anchor is the A30 primary; without it
        # the pair does not hold and structural admission (kernels + device)
        # is the only path — absent on this host.
        with self.assertRaisesRegex(ValueError, 'Triton hybrid kernels'):
            self.q4_identity({'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
                              'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
                              'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                              'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
                              'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '22528,13312'},
                             primary_device='NVIDIA A2')

    CHAIN_ENV = {'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
                 'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
                 'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
                 'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
                 'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '22528',
                 'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN': '1'}

    def test_q4_chain_admits_remote_tail_rank_at_qualified_context(self):
        # #536 gate 4: a single-GPU remote tail rank (no A30 primary) is the
        # qualified three-device chain topology; the explicit chain opt-in
        # admits it up to the qualified 2048 context cell.
        self.q4_identity(self.CHAIN_ENV,
                         primary_device='NVIDIA GeForce RTX 3090')

    def test_q4_chain_rejected_without_chain_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'Triton hybrid kernels'):
            self.q4_identity({key: value for key, value in self.CHAIN_ENV.items()
                              if key != 'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN'},
                             primary_device='NVIDIA GeForce RTX 3090')

    def test_q4_chain_context_bound_is_the_qualified_cell(self):
        import damendaemon.hybrid_q4_policy as module
        env = dict(self.CHAIN_ENV, DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB='22528')
        with patch.dict('os.environ', env, clear=True), \
                patch.object(module, 'selected_accelerator', return_value='cuda'):
            module.validate(self.config, 2048)
            with self.assertRaisesRegex(ValueError, 'context <=2048'):
                module.validate(self.config, 2049)
        without_chain = {key: value for key, value in env.items()
                         if key != 'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN'}
        with patch.dict('os.environ', without_chain, clear=True), \
                patch.object(module, 'selected_accelerator', return_value='cuda'):
            with self.assertRaisesRegex(ValueError, 'context <=512'):
                module.validate(self.config, 513)

    def test_q4_chain_keeps_standard_guards(self):
        # The chain opt-in lifts only the A30-primary anchor and the context
        # bound; precision (and every other standard guard) still applies.
        for dropped in ('DAMENDAEMON_HYBRID_PRECISION',
                        'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4',
                        'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'):
            env = {key: value for key, value in self.CHAIN_ENV.items() if key != dropped}
            env['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = '22528'
            with self.assertRaisesRegex(ValueError, 'opt-in|compensated'):
                self.q4_identity(env, primary_device='NVIDIA GeForce RTX 3090')

    def test_q4_chain_admits_compensated_select_at_qualified_context(self):
        # hybrid_precision.select carried its own context<=512 pin; under the
        # chain opt-in the qualified 2048 cell is admitted for Q4.
        from types import SimpleNamespace
        from damendaemon.hybrid_precision import select
        model = SimpleNamespace(quantization='q4_0', config=self.config,
                                max_context=2048, vision_weights=None)
        env = dict(self.CHAIN_ENV, DAMENDAEMON_DEVICE='cuda')
        with patch.dict('os.environ', env, clear=True), \
                patch('damendaemon.hybrid_precision.selected_accelerator', return_value='cuda'):
            self.assertEqual(select(model, False, 1, 'triton'), 'compensated')
        beyond = SimpleNamespace(quantization='q4_0', config=self.config,
                                  max_context=2049, vision_weights=None)
        with patch.dict('os.environ', env, clear=True), \
                patch('damendaemon.hybrid_precision.selected_accelerator', return_value='cuda'):
            with self.assertRaisesRegex(ValueError, 'context <=2048'):
                select(beyond, False, 1, 'triton')

    def test_pipeline_context_cap_is_32768(self):
        # #536 gate 3: the context cap moved from 2048 to 32768 (bounded by
        # the #70-planner evidence, not device memory); the checkpoint's own
        # max_position_embeddings still binds above that. One patched-env
        # block covers all three calls: outside it the pipeline opt-in is
        # unset and validate returns before the context check.
        env = {'DAMENDAEMON_DEVICE': 'cuda',
               'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE': '1',
               'DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT': '1',
               'DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB': '2048,2048',
               'DAMENDAEMON_HYBRID_PRECISION': 'compensated'}
        with patch.dict('os.environ', env, clear=True):
            policy_validate(self.config, 'q4_0', 32768, 1, False)
            with self.assertRaisesRegex(ValueError, 'context <=32768'):
                policy_validate(self.config, 'q4_0', 32769, 1, False)


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

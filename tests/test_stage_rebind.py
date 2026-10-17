"""Real tensor copy/identity tests; GPU graph and peer proof remain separate."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from damendaemon.layer_stage import shard_model
from damendaemon.stage_rebind import compatible_shard, copy_stage_weights, commit_stage_binding


class StageRebindTests(unittest.TestCase):
    def fixture(self):
        model = SimpleNamespace(
            config=dict(model_type='qwen2',num_hidden_layers=2),
            quantization='q8_0',max_context=512,
            weights={'0.out':torch.zeros(2,32,dtype=torch.int8),
                     '1.out':torch.ones(2,32,dtype=torch.int8)},
            scales={'0.out':torch.ones(2,1,dtype=torch.float16),
                    '1.out':torch.ones(2,1,dtype=torch.float16)})
        shard = shard_model(model,0,1)
        weights = {name:t.clone() for name,t in shard.weights.items()}
        stage = SimpleNamespace(layer_range=(0,1),model=shard,c=shard.config,
            w=weights,lengths=[11,0],q4_linear=None,
            quant_linear=SimpleNamespace(scales={id(weights[name]):t.clone() for name,t in shard.scales.items()}))
        replacement = copy.copy(model)
        replacement.weights = {name:t+3 for name,t in model.weights.items()}
        replacement.scales = {name:t*2 for name,t in model.scales.items()}
        return stage,replacement

    def test_copy_preserves_addresses_and_waits_to_publish_identity(self):
        stage,model = self.fixture()
        previous = stage.model
        weight = stage.w['0.out']
        scale = stage.quant_linear.scales[id(weight)]
        pointers = weight.data_ptr(),scale.data_ptr()
        shard = compatible_shard(stage,model)
        self.assertIsNotNone(shard)
        copy_stage_weights(stage,shard)
        self.assertIs(stage.model,previous)
        self.assertEqual(stage.lengths,[11,0])
        self.assertTrue(torch.equal(weight,model.weights['0.out']))
        self.assertTrue(torch.equal(scale,model.scales['0.out']))
        self.assertEqual(pointers,(weight.data_ptr(),scale.data_ptr()))
        commit_stage_binding(stage,shard,2)
        self.assertIs(stage.model,shard)
        self.assertEqual(stage.lengths,[0,0])

    def test_repack_failure_leaves_identity_unpublished(self):
        stage,model = self.fixture()
        previous = stage.model
        stage.q4_linear = SimpleNamespace(refresh=Mock(side_effect=RuntimeError('partial repack')))
        with self.assertRaisesRegex(RuntimeError,'partial repack'):
            copy_stage_weights(stage,compatible_shard(stage,model))
        self.assertIs(stage.model,previous)
        self.assertEqual(stage.lengths,[11,0])
        # Buffers may already be changed; the owner MUST poison its session.
        self.assertTrue(torch.equal(stage.w['0.out'],model.weights['0.out']))

    def test_configuration_shape_dtype_or_alias_changes_reject_before_copy(self):
        for change in ('config','context','precision','shape','dtype','aliases'):
            with self.subTest(change=change):
                stage,model = self.fixture()
                if change=='config':model.config={**model.config,'rms_norm_eps':0.1}
                if change=='context':model.max_context=256
                if change=='precision':model.quantization='q4_0'
                if change=='shape':model.weights['0.out']=torch.zeros(3,32,dtype=torch.int8)
                if change=='dtype':model.weights['0.out']=model.weights['0.out'].float()
                if change=='aliases':
                    stage.w['0.other']=stage.w['0.out']
                    stage.model.weights['0.other']=stage.model.weights['0.out']
                    model.weights['0.other']=model.weights['0.out'].clone()
                self.assertIsNone(compatible_shard(stage,model))
                self.assertEqual(stage.lengths,[11,0])
                self.assertEqual(torch.count_nonzero(stage.w['0.out']).item(),0)

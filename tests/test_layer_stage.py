"""Host ownership invariants; GPU correctness is qualified by real traces."""
import unittest
from types import SimpleNamespace
import torch
from scripts.experiments.layer_stage import shard_model


class LayerStageTests(unittest.TestCase):
    def model(self):
        weights={'embed':torch.zeros(4,32),'head':torch.zeros(4,32,dtype=torch.int8),'norm':torch.ones(32)}
        weights.update({f'{i}.out':torch.zeros(4,32,dtype=torch.int8) for i in range(4)})
        scales={name:torch.ones(4,1,dtype=torch.float16) for name in weights if name!='embed' and name!='norm'}
        return SimpleNamespace(config={'model_type':'qwen2','num_hidden_layers':4},quantization='q8_0',weights=weights,scales=scales)

    def test_q8_scales_follow_selected_weights_without_mutating_source(self):
        model=self.model(); before=dict(model.weights); scales=dict(model.scales)
        stage=shard_model(model,2,4)
        self.assertEqual(set(stage.weights),{'0.out','1.out','head','norm'})
        self.assertEqual(set(stage.scales),{'0.out','1.out','head'})
        self.assertIs(stage.weights['0.out'],model.weights['2.out'])
        self.assertIs(stage.scales['0.out'],model.scales['2.out'])
        self.assertEqual(model.weights,before);self.assertEqual(model.scales,scales)
        self.assertEqual(model.config['num_hidden_layers'],4)
        expected=sum(t.numel()*t.element_size() for t in (*stage.weights.values(),*stage.scales.values()))
        self.assertEqual(stage.bytes,expected)
        first=shard_model(model,0,2)
        self.assertIn('embed',first.weights);self.assertNotIn('head',first.weights)
        self.assertNotIn('head',first.scales)

    def test_unqualified_quantization_and_invalid_layer_bounds_rejected(self):
        model=self.model()
        for start,end in ((-1,2),(0,5),(2,2),(3,1)):
            with self.assertRaises(ValueError):shard_model(model,start,end)
        model.quantization='q4_0'
        with self.assertRaises(ValueError):shard_model(model,0,2)

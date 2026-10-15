import unittest
from unittest.mock import patch
import torch
from dllm.cuda_deepseek_placement import place_cuda_deepseek,_tensors,_build
from tests.test_packed_model import fixture
from tests.test_packed_attention import prepare


class DeepSeekPlacementTests(unittest.TestCase):
    def test_budget_policy_and_late_metadata_fail_before_copy(self):
        source,_,_=fixture()
        with patch('dllm.cuda_deepseek_placement._copy',side_effect=AssertionError('copy before admission')):
            for options in ({'device':'cuda','max_bytes':10**9},{'device':'cuda:0','max_bytes':1},
                            {'device':'cuda:0','max_bytes':10**9,'workspace_bytes':1}):
                with self.assertRaises(ValueError):place_cuda_deepseek(source,**options)
            source.blocks[-1].weights['attn_norm.weight']=torch.ones(1)
            with self.assertRaises(ValueError):place_cuda_deepseek(source,device='cuda:0',max_bytes=10**9)

    def test_cpu_recipe_preserves_expert_aliases_config_and_projection_rows(self):
        source,_,ids=fixture();source.config={'nested':{'value':1}}
        for block in source.blocks:block.attention_weights,_=prepare(block.attention_weights)
        from tests.test_packed_projection import fixture as projection_fixture
        from dllm.prepared_projection import PackedProjection
        _,raw,scales,_=projection_fixture('e4m3fn',7,32)
        source.blocks[0].attention_weights['wq_a.weight']=PackedProjection(raw,scales,format='e4m3fn').rows(1,5)
        tensors=_tensors(source)
        rebuilt=_build(source,tensors,cuda=False,workspace_bytes=256*1024**2,max_output_bytes=64*1024**2)
        self.assertIs(rebuilt.blocks[0].packed_experts[0],rebuilt.blocks[1].packed_experts[0])
        self.assertIsNot(rebuilt.config,source.config);rebuilt.config['nested']['value']=2
        self.assertEqual(source.config['nested']['value'],1)
        expected,a=source.prefill(ids[:,:3]);actual,b=rebuilt.prefill(ids[:,:3])
        try:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        finally:a.close();b.close()
        projection=rebuilt.blocks[0].attention_weights['wq_a.weight']
        self.assertEqual((projection.start,projection.stop),(1,5))
        self.assertIs(projection.weight,raw)

    def test_non_fp32_policy_rejected_before_copy(self):
        source,_,_=fixture();source.embedding=source.embedding.half()
        with patch('dllm.cuda_deepseek_placement._copy',side_effect=AssertionError('copy before admission')):
            with self.assertRaisesRegex(ValueError,'FP32'):place_cuda_deepseek(source,device='cuda:0',max_bytes=10**9)

    def test_loader_sized_expert_budget_reserves_outer_routing_scratch(self):
        from dllm.cuda_deepseek_placement import _policy,_inner_workspace
        source,_,_=fixture()
        for block in source.blocks:
            block.workspace_bytes=max(e.workspace_bytes for e in (*block.packed_experts,block.packed_shared))
            self.assertGreater(_inner_workspace(block),block.workspace_bytes)
        _policy(source,256*1024**2,64*1024**2)

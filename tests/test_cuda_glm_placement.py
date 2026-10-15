import unittest
from unittest.mock import patch
import torch
from dllm.cuda_glm_placement import place_cuda_glm,_tensors,_build
from tests.test_glm_packed_model import model_pair


class CudaGlmPlacementTests(unittest.TestCase):
    def test_whole_budget_and_late_invalid_component_before_copy(self):
        source,_=model_pair();tensors=_tensors(source)
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        with patch('dllm.cuda_glm_placement._copy',side_effect=AssertionError('copied')) as copy:
            with self.assertRaises(ValueError):place_cuda_glm(source,device='cuda:0',max_bytes=size-1)
            source.blocks[-1].post_norm=torch.ones(1)
            with self.assertRaises(ValueError):place_cuda_glm(source,device='cuda:0',max_bytes=size+100)
            copy.assert_not_called()

    def test_alias_accounting_and_preserved_cpu_recipe(self):
        source,_=model_pair();source.head=source.embedding
        source.config={'max_position_embeddings':64,'eos_token_id':[1,2,3]}
        tensors=_tensors(source)
        self.assertIs(tensors[id(source.head)],source.embedding)
        rebuilt=_build(source,tensors,cuda=False)
        self.assertIs(rebuilt.head,rebuilt.embedding)
        self.assertEqual(rebuilt.config,source.config)
        rebuilt.config['eos_token_id'].append(4)
        self.assertEqual(source.config['eos_token_id'],[1,2,3])
        tokens=torch.tensor([[1,2,3]])
        left,a=source.prefill(tokens,capacity=3,all_logits=True)
        right,b=rebuilt.prefill(tokens,capacity=3,all_logits=True)
        try:torch.testing.assert_close(left,right,rtol=0,atol=0)
        finally:a.close();b.close()
        self.assertEqual(rebuilt.blocks[1].feed_forward.route_scale,source.blocks[1].feed_forward.route_scale)
        self.assertEqual(rebuilt.blocks[0].attention.budgets,source.blocks[0].attention.budgets)

    def test_copy_failure_leaves_source_usable(self):
        source,_=model_pair();before=source.embedding.clone()
        with patch('dllm.cuda_glm_placement._copy',side_effect=MemoryError('allocation')):
            with self.assertRaises(MemoryError):place_cuda_glm(source,device='cuda:0',max_bytes=10**6)
        torch.testing.assert_close(source.embedding,before,rtol=0,atol=0)
        _,state=source.prefill(torch.tensor([[1]]),capacity=1);state.close()
        for device in ('cpu','cuda'):
            with self.assertRaises(ValueError):place_cuda_glm(source,device=device,max_bytes=10**6)


if __name__=='__main__':unittest.main()

import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_glm_placement import place_cuda_glm,_tensors,_build
from tests.test_glm_packed_model import model_pair


class CudaGlmPlacementTests(unittest.TestCase):
    def test_whole_budget_and_late_invalid_component_before_copy(self):
        source,_=model_pair();tensors=_tensors(source)
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        with patch('damendaemon.cuda_glm_placement._copy',side_effect=AssertionError('copied')) as copy:
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
        with patch('damendaemon.cuda_glm_placement._copy',side_effect=MemoryError('allocation')):
            with self.assertRaises(MemoryError):place_cuda_glm(source,device='cuda:0',max_bytes=10**6)
        torch.testing.assert_close(source.embedding,before,rtol=0,atol=0)
        _,state=source.prefill(torch.tensor([[1]]),capacity=1);state.close()
        for device in ('cpu','cuda'):
            with self.assertRaises(ValueError):place_cuda_glm(source,device=device,max_bytes=10**6)

    def test_composition_options_rejected_before_copy(self):
        source,_=model_pair();tensors=_tensors(source)
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        for kwargs in ({'device_routing':'yes'},{'composition':'fused'},{'composition':'BATCHED'}):
            with patch('damendaemon.cuda_glm_placement._copy',side_effect=AssertionError('copied')):
                with self.assertRaises(ValueError):
                    place_cuda_glm(source,device='cuda:0',max_bytes=size+100,**kwargs)
                # _copy asserted not called by raising on first use above.

    def test_composition_options_threaded_into_cuda_ffn(self):
        source,_=model_pair();tensors=_tensors(source)
        with patch('damendaemon.cuda_glm_placement.CudaCachedGlmAttention'), \
             patch('damendaemon.cuda_glm_placement.CudaPackedExpert'), \
             patch('damendaemon.cuda_glm_placement.CudaPackedMatrix'), \
             patch('damendaemon.cuda_glm_placement.CudaPackedProjection'), \
             patch('damendaemon.cuda_glm_placement.CudaGlmBlock'), \
             patch('damendaemon.cuda_glm_placement.CudaGlmModel'), \
             patch('damendaemon.cuda_glm_placement.CudaPackedGlmFeedForward') as ffn_type:
            _build(source,tensors,cuda=True,device_routing=True,composition='batched')
            kwargs=ffn_type.call_args.kwargs
            self.assertEqual(kwargs['device_routing'],True)
            self.assertEqual(kwargs['composition'],'batched')
            cuda_calls=ffn_type.call_count
            _build(source,tensors,cuda=False,device_routing=True,composition='batched')
            # The CPU revalidation build uses the real CPU FFN class, not the
            # CUDA mock, and receives no dispatch options.
            self.assertEqual(ffn_type.call_count,cuda_calls)
            rebuilt=_build(source,tensors,cuda=False)
            self.assertEqual(rebuilt.blocks[0].feed_forward.__class__.__name__,'PackedGlmFeedForward')


if __name__=='__main__':unittest.main()

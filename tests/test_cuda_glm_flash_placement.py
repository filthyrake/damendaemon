import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_glm_flash_placement import place_cuda_flash,_tensors,_build
from tests.test_glm_flash_model import fixture as model_fixture
from tests.test_glm_flash_ffn import packed_ffn
from tests.test_glm_flash_sparse import fixture as sparse_fixture


def fixture():
    source=model_fixture();source.blocks[1].attention=sparse_fixture(True)[0]
    for block in source.blocks:block.feed_forward=packed_ffn(block.feed_forward.weights,False,.5)[0]
    return source


class CudaFlashPlacementTests(unittest.TestCase):
    def test_aggregate_and_late_invalid_before_copy(self):
        source=fixture();tensors=_tensors(source);size=sum(v.numel()*v.element_size() for v in tensors.values())
        with patch('damendaemon.cuda_glm_flash_placement._copy',side_effect=AssertionError('copied')) as copy:
            with self.assertRaises(ValueError):place_cuda_flash(source,device='cuda:0',max_bytes=size-1)
            source.blocks[-1].post_norm=torch.ones(1)
            with self.assertRaises(ValueError):place_cuda_flash(source,device='cuda:0',max_bytes=size+100)
            copy.assert_not_called()

    def test_recipe_alias_config_limits_and_logits(self):
        source=fixture();source.head=source.embedding;source.config={'eos_token_id':[1,2]}
        source.blocks[0].max_output_bytes=123456
        source.blocks[0].attention_hyper.iterations=3
        rebuilt=_build(source,_tensors(source),cuda=False)
        self.assertIs(rebuilt.embedding,rebuilt.head)
        self.assertEqual(rebuilt.blocks[0].max_output_bytes,123456)
        self.assertEqual(rebuilt.blocks[0].attention_hyper.iterations,3)
        self.assertEqual(rebuilt.blocks[0].feed_forward.activation_limit,.5)
        rebuilt.config['eos_token_id'].append(3);self.assertEqual(source.config['eos_token_id'],[1,2])
        tokens=torch.tensor([[1,2,3]])
        left,a=source.prefill(tokens,capacity=3,all_logits=True)
        right,b=rebuilt.prefill(tokens,capacity=3,all_logits=True)
        try:torch.testing.assert_close(left,right,atol=0,rtol=0)
        finally:a.close();b.close()

    def test_copy_failure_preserves_source_and_explicit_device(self):
        source=fixture();before=source.embedding.clone()
        with patch('damendaemon.cuda_glm_flash_placement._copy',side_effect=MemoryError('allocation')):
            with self.assertRaises(MemoryError):place_cuda_flash(source,device='cuda:0',max_bytes=10**6)
        torch.testing.assert_close(source.embedding,before,atol=0,rtol=0)
        _,state=source.prefill(torch.tensor([[1]]),capacity=1);state.close()
        for device in ('cpu','cuda'):
            with self.assertRaises(ValueError):place_cuda_flash(source,device=device,max_bytes=10**6)

    def test_composition_options_threaded_into_cuda_ffn(self):
        source=fixture()
        tensors=_tensors(source)
        with patch('damendaemon.cuda_glm_flash_placement.CudaFlashLinearAttention'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaFlashSparseAttention'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaFlashIndexer'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaFlashHyperConnection'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaPackedExpert'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaPackedMatrix'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaPackedProjection'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaFlashBlock'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaFlashModel'), \
             patch('damendaemon.cuda_glm_flash_placement.CudaPackedFlashFeedForward') as ffn_type:
            _build(source,tensors,cuda=True,device_routing=True,composition='batched')
            kwargs=ffn_type.call_args.kwargs
            self.assertEqual(kwargs['device_routing'],True)
            self.assertEqual(kwargs['composition'],'batched')
            cuda_calls=ffn_type.call_count
            _build(source,tensors,cuda=False,device_routing=True,composition='batched')
            # The CPU revalidation build uses the real CPU FFN class, not the
            # CUDA mock, and receives no dispatch options.
            self.assertEqual(ffn_type.call_count,cuda_calls)

import unittest
from unittest.mock import patch
from damendaemon.cuda_glm_flash_model import CudaFlashModel,CudaFlashBlock,CudaPackedFlashFeedForward
from tests.test_glm_flash_model import fixture


class CudaFlashModelAdmissionTests(unittest.TestCase):
    def test_cpu_model_rejected_before_core(self):
        cpu=fixture()
        with patch('damendaemon.cuda_glm_flash_model._FlashModel.__init__',side_effect=AssertionError('core entered')):
            with self.assertRaisesRegex(ValueError,'CUDA tensors'):CudaFlashModel(cpu.embedding,cpu.blocks,cpu.norm,cpu.head)

    def test_cpu_block_rejected(self):
        block=fixture().blocks[0]
        with self.assertRaisesRegex(ValueError,'CUDA tensors'):
            CudaFlashBlock(block.attention,block.feed_forward,block.attention_hyper,block.ffn_hyper,block.input_norm,block.post_norm)

    def test_clipping_required(self):
        for value in (None,True,0,float('inf'),float('nan')):
            with self.assertRaisesRegex(ValueError,'activation limit'):CudaPackedFlashFeedForward(None,activation_limit=value)

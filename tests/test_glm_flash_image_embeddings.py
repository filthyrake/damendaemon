import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_image_embeddings import image_embeddings
from damendaemon.glm_flash_images import prepare_images
from damendaemon.glm_flash_vision import FlashVision,vision_specs
from tests import test_glm_flash_model,test_glm_flash_block,test_glm_flash_vision


def control(values,model):
    hidden=values.unsqueeze(-2).expand(-1,-1,model.streams,-1)
    for sparse,block in zip((False,True,False),model.blocks):hidden=test_glm_flash_block.control(hidden,block,sparse)
    value=hidden.mean(-2)
    return (value/(value.square().mean(-1,keepdim=True)+model.epsilon).sqrt()*model.norm)@model.head.T


class FlashImageEmbeddingTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        self.model=test_glm_flash_model.fixture()

    def test_mixed_rows_order_chunked_images_and_ordinary_decode(self):
        tokens=torch.tensor([[1,10,10,2,10],[3,4,5,6,7]])
        features=[torch.arange(12).reshape(3,4).float()/10,torch.empty(0,4)]
        original=tokens.clone();values=image_embeddings(self.model,tokens,features,image_token_id=10)
        expected=self.model.embedding[tokens].clone();expected[0,[1,2,4]]=features[0]
        torch.testing.assert_close(values,expected,rtol=0,atol=0);torch.testing.assert_close(tokens,original)
        logits=control(expected,self.model)
        actual,state=self.model.prefill_embeddings(values[:,:2],capacity=6,all_logits=True);self.addCleanup(state.close)
        torch.testing.assert_close(actual,logits[:,:2],rtol=6e-5,atol=3e-6)
        actual=self.model.forward_embeddings(values[:,2:],state,position=2,all_logits=True)
        torch.testing.assert_close(actual,logits[:,2:],rtol=6e-5,atol=3e-6)
        next_tokens=torch.tensor([[8],[9]])
        actual=self.model.forward(next_tokens,state,position=5)
        complete=torch.cat((expected,self.model.embedding[next_tokens]),1)
        torch.testing.assert_close(actual,control(complete,self.model)[:,-1],rtol=6e-5,atol=3e-6)
        self.assertEqual(state.position,6)

    def test_rgb_to_vision_to_language_fixture(self):
        pixels=torch.full((112,112,3),127,dtype=torch.uint8);patches,grids=prepare_images([pixels])
        c,_=test_glm_flash_vision.fixture();c.update(patch_size=14,out_hidden_size=4,projection_intermediate_size=8)
        torch.manual_seed(110);weights={name:torch.randn(shape)*.02 for name,shape in vision_specs(c).items()}
        features=FlashVision(c,weights).forward(patches,grids)
        tokens=torch.tensor([[1]+[10]*features.shape[0]+[2]])
        values=image_embeddings(self.model,tokens,[features],image_token_id=10)
        actual,state=self.model.prefill_embeddings(values,capacity=tokens.shape[1]);self.addCleanup(state.close)
        torch.testing.assert_close(actual,control(values,self.model)[:,-1],rtol=6e-5,atol=3e-6)

    def test_bad_features_reject_before_embedding_or_cache_mutation(self):
        tokens=torch.tensor([[10,10]])
        for value in (torch.zeros(1,4),torch.zeros(2,5),torch.zeros(2,4).half(),torch.full((2,4),float('nan'))):
            with patch('damendaemon.glm_flash_image_embeddings.F.embedding',side_effect=AssertionError('allocated')):
                with self.assertRaises(ValueError):image_embeddings(self.model,tokens,[value],image_token_id=10)
        for token_id in (True,-1,13):
            with self.assertRaises(ValueError):image_embeddings(self.model,tokens,[torch.zeros(2,4)],image_token_id=token_id)
        state=self.model.new_state(1,3);self.addCleanup(state.close)
        for value in (torch.zeros(1,1,5),torch.zeros(1,1,4).half(),torch.full((1,1,4),float('inf'))):
            with self.assertRaises(ValueError):self.model.forward_embeddings(value,state,position=0)
        self.assertEqual(state.position,0);self.assertFalse(state.poisoned)
        with patch.object(self.model,'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaises(ValueError):self.model.prefill_embeddings(torch.zeros(1,4,4),capacity=3)
        self.model.max_output_bytes=1
        with patch.object(self.model,'new_state',side_effect=AssertionError('allocated')):
            with self.assertRaises(ValueError):self.model.prefill_embeddings(torch.zeros(1,1,4),capacity=1)

    def test_partial_execution_poison_and_owned_prefill_cleanup(self):
        values=torch.zeros(1,2,4);state=self.model.new_state(1,2);self.addCleanup(state.close)
        with patch.object(self.model,'_logits',side_effect=RuntimeError('output failed')):
            with self.assertRaises(RuntimeError):self.model.forward_embeddings(values,state,position=0)
        self.assertTrue(state.poisoned);self.assertEqual(state.position,0)
        state=self.model.new_state(1,2)
        with patch.object(self.model,'new_state',return_value=state),patch.object(self.model,'_logits',side_effect=RuntimeError('output failed')):
            with self.assertRaises(RuntimeError):self.model.prefill_embeddings(values,capacity=2)
        self.assertTrue(state.closed);self.assertIsNone(state.owner)

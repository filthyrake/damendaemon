from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tokenizers import Tokenizer,AddedToken
from dllm.glm_flash_image_generation import FlashImageReferenceGenerator
from dllm.glm_flash_image_text import FlashImageText,BEGIN_IMAGE,IMAGE,END_IMAGE
from dllm.glm_flash_vision import FlashVision,vision_specs
from dllm.glm_text import EOS,END_THINK
from tests import test_glm_generation,test_glm_flash_generation,test_glm_flash_vision,test_glm_flash_model


def text_fixture(path):
    test_glm_generation.text_fixture(path);tokenizer=Tokenizer.from_file(str(path/'tokenizer.json'))
    tokenizer.add_special_tokens([AddedToken(marker,special=True) for marker in (BEGIN_IMAGE,IMAGE,END_IMAGE)])
    tokenizer.save(str(path/'tokenizer.json'))
    return FlashImageText(path,vocab_size=tokenizer.get_vocab_size()+24)


def vision(width):
    c,_=test_glm_flash_vision.fixture();c.update(out_hidden_size=width,projection_intermediate_size=8,patch_size=14)
    torch.manual_seed(113);weights={name:torch.randn(shape)*.02 for name,shape in vision_specs(c).items()}
    return FlashVision(c,weights)


class ScriptedImageModel(test_glm_flash_generation.ScriptedFlash):
    def __init__(self,text,completion):
        super().__init__(text,completion);self.streams=2;self.max_output_bytes=64*1024**2
        self.image_id=text.image_token_id
    def logits(self,index):
        result=super().logits(index);result[0,self.image_id]=1000;return result
    def prefill_embeddings(self,values,**options):
        self.last_values=values.clone();return super().prefill(values,**options)


class FlashImageGenerationTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.text=text_fixture(Path(temp.name))
        self.messages=[dict(role='user',content=[dict(type='text',text='Hi'),dict(type='image')])]
        self.images=[torch.full((112,112,3),127,dtype=torch.uint8)]

    def runner(self,completion='analysis'+END_THINK+'answer'+EOS):
        model=ScriptedImageModel(self.text,completion);return FlashImageReferenceGenerator(model,self.text,vision(1),max_context=128)

    def test_composed_response_usage_media_suppression_and_release(self):
        runner=self.runner();events=list(runner.generate_images(self.messages,self.images,max_new_tokens=40))
        self.assertEqual(''.join(e.get('content','') for e in events),'answer')
        self.assertEqual(events[-1]['finish_reason'],'stop');self.assertTrue(runner.model.states[-1].closed)
        prompt=self.text.image_chat_tokens(self.messages,[[1,8,8]])
        self.assertEqual(events[-1]['usage']['prompt_tokens'],len(prompt))
        self.assertEqual(runner.model.last_values.shape,(1,len(prompt),1))

    def test_actual_fixture_language_generation_and_plain_path_preserved(self):
        model=test_glm_flash_model.fixture();model.embedding=torch.randn(self.text.vocab_size,4)*.1
        model.head=torch.zeros(self.text.vocab_size,4)
        runner=FlashImageReferenceGenerator(model,self.text,vision(4),max_context=128)
        events=list(runner.generate_images(self.messages,self.images,max_new_tokens=2))
        self.assertEqual(events[-1]['finish_reason'],'length');self.assertEqual(events[-1]['usage']['completion_tokens'],2)
        self.assertFalse(runner.failed_cleanups);self.assertFalse(model.failed_cleanups)
        plain=list(runner.generate([dict(role='user',content='Hi')],max_new_tokens=2))
        self.assertEqual(plain[-1]['finish_reason'],'length')

    def test_cancel_and_primary_error_retain_failed_cleanup_for_retry(self):
        runner=self.runner('abcdef');iterator=runner.generate_images(self.messages,self.images,max_new_tokens=5)
        next(iterator);iterator.close();self.assertTrue(runner.model.states[-1].closed)
        iterator=runner.generate_images(self.messages,self.images,max_new_tokens=5);next(iterator);state=runner.model.states[-1]
        with patch.object(runner.model,'forward',side_effect=RuntimeError('primary')),patch.object(state,'close',side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(RuntimeError,'primary') as caught:list(iterator)
            self.assertTrue(caught.exception.__notes__);self.assertEqual(runner.failed_cleanups,[state])
            with patch.object(runner.vision,'forward',side_effect=AssertionError('vision before cleanup')):
                with self.assertRaisesRegex(RuntimeError,'cleanup'):list(runner.generate_images(self.messages,self.images,max_new_tokens=2))
        runner.cleanup();self.assertTrue(state.closed)

    def test_bad_counts_and_context_fail_before_vision_or_cache(self):
        runner=self.runner()
        with patch.object(runner.vision,'forward',side_effect=AssertionError('vision')),patch.object(runner.model,'prefill_embeddings',side_effect=AssertionError('cache')):
            for images,limit in (([],2),(self.images,128),(self.images,True)):
                with self.assertRaises(ValueError):list(runner.generate_images(self.messages,images,max_new_tokens=limit))
        with self.assertRaises(ValueError):FlashImageReferenceGenerator(runner.model,self.text,vision(4),max_context=128)
        with patch.object(runner.vision,'forward',side_effect=RuntimeError('vision failed')):
            with self.assertRaisesRegex(RuntimeError,'vision failed'):list(runner.generate_images(self.messages,self.images,max_new_tokens=2))
        self.assertFalse(runner.model.states)

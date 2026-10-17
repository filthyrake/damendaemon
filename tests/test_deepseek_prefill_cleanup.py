from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from damendaemon.floating_attention import AttentionState
from damendaemon.deepseek_reference_loader import load_reference_model,load_reference_stage
from damendaemon.deepseek_generation import ReferenceGenerator
from damendaemon.reference_serving import ReferenceRuntime
from tests.test_floating_block import fixture as block_fixture
from tests.test_deepseek_reference_loader import fixture
from tests.test_deepseek_generation import text_fixture


class DeepSeekPrefillCleanupTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def test_block_retains_failed_attention_and_original_error_before_retry(self):
        block,hidden,tokens=block_fixture()
        with patch.object(block,'_ffn',side_effect=ValueError('original FFN')),patch.object(AttentionState,'close',side_effect=OSError('release')):
            with self.assertRaisesRegex(ValueError,'original FFN') as error:block.prefill(hidden[:,:3],tokens[:,:3],max_tokens=8)
            self.assertTrue(error.exception.__notes__);self.assertEqual(len(block.failed_prefills),1)
            state=block.failed_prefills[0];self.assertIsNotNone(state.cache)
            with patch('damendaemon.floating_block.attention_prefill',side_effect=AssertionError('allocate')):
                with self.assertRaisesRegex(OSError,'release'):block.prefill(hidden[:,:3],tokens[:,:3],max_tokens=8)
        block.cleanup_failed_prefills();self.assertFalse(block.failed_prefills);self.assertTrue(state.closed)
        _,state=block.prefill(hidden[:,:3],tokens[:,:3],max_tokens=8);state.close()

    def test_whole_model_runner_and_http_status_keep_failed_block_owner_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);text=text_fixture(path);fixture(path,vocab_size=text.vocab_size)
            model=load_reference_model(path);runner=ReferenceGenerator(model,text,max_context=512)
            runtime=ReferenceRuntime(runner);self.addCleanup(runtime.close)
            with patch.object(model.blocks[0],'_ffn',side_effect=ValueError('original FFN')),patch.object(AttentionState,'close',side_effect=OSError('release')):
                with self.assertRaisesRegex(ValueError,'original FFN'):model.prefill(torch.tensor([[1,2]]),max_tokens=8)
                self.assertFalse(model.failed_prefills);self.assertTrue(model.prefill_cleanup_pending)
                self.assertTrue(runtime.cleanup_pending());self.assertTrue(runtime.manage('status')['cleanup_pending'])
                with self.assertRaisesRegex(OSError,'release'):runner.cleanup()
                with self.assertRaisesRegex(OSError,'release'):runtime.close()
            runtime.close();self.assertTrue(runtime.closed);self.assertFalse(model.prefill_cleanup_pending)

    def test_partial_stage_prefill_retains_previous_and_current_cleanup_owners(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);fixture(path);stage=load_reference_stage(path,0,2);self.addCleanup(stage.cleanup)
            tokens=torch.tensor([[1,2,3]])
            with patch.object(stage.blocks[-1],'_ffn',side_effect=ValueError('later FFN')),patch.object(AttentionState,'close',side_effect=OSError('release')):
                with self.assertRaisesRegex(ValueError,'later FFN'):stage.prefill(tokens,max_tokens=8)
                self.assertTrue(stage.failed_prefills);self.assertTrue(stage.blocks[-1].failed_prefills)
                self.assertTrue(stage.prefill_cleanup_pending)
                with patch.object(stage.blocks[0],'prefill',side_effect=AssertionError('allocate')):
                    with self.assertRaisesRegex(OSError,'release'):stage.prefill(tokens,max_tokens=8)
            stage.cleanup();self.assertFalse(stage.prefill_cleanup_pending)
            _,state=stage.prefill(tokens,max_tokens=8);state.close()

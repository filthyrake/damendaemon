"""Parser storage must be released even when decoding fails before parser entry."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dllm.deepseek_generation import TokenStream
from dllm.deepseek_tools import START
from dllm.glm_flash_tool_output import FlashToolTokenStream
from dllm.glm_flash_tool_text import FlashToolText
from dllm.glm_flash_tools import FlashTools
from dllm.glm_text import END_THINK
from tests.test_deepseek_generation import text_fixture as deepseek_fixture
from tests.test_glm_flash_generation import text_fixture as flash_fixture
from tests.test_glm_flash_tools import tools


class ToolStreamCleanupTests(unittest.TestCase):
    def test_decoder_failures_close_both_tool_buffers(self):
        for family in ('deepseek','flash'):
            for failure in ('token','feed','finish'):
                with self.subTest(family=family,failure=failure),TemporaryDirectory() as temp:
                    path=Path(temp)
                    if family=='deepseek':
                        text=deepseek_fixture(path)
                        stream=TokenStream(text,allow_tool_calls=True)
                        prefix='\n\n'+START+'x'*100
                        buffer_name='tool_buffer'
                    else:
                        base=flash_fixture(path)
                        text=FlashToolText(path,vocab_size=base.vocab_size)
                        stream=FlashToolTokenStream(text,FlashTools(tools()))
                        prefix=END_THINK+'<tool_call>f'+'x'*100
                        buffer_name='buffer'
                    for token in text.raw_tokens(prefix):stream.feed(token)
                    buffer=getattr(stream.parser,buffer_name)
                    self.assertFalse(buffer.closed)
                    if failure=='token':
                        with self.assertRaises(ValueError):stream.feed(-1)
                    else:
                        error=RuntimeError('decoder failed')
                        with patch.object(stream.decoder,failure,side_effect=error):
                            with self.assertRaises(RuntimeError) as caught:
                                stream.feed(0) if failure=='feed' else stream.finish()
                        self.assertIs(caught.exception,error)
                    self.assertTrue(stream.failed)
                    self.assertTrue(buffer.closed)
                    self.assertTrue(stream.parser.closed)
                    with self.assertRaises(RuntimeError):stream.feed(0)
                    with self.assertRaises(RuntimeError):stream.finish()
                    stream.close()  # Idempotent even after failure.

    def test_ordinary_glm_close_discards_pending_marker_and_blocks_reuse(self):
        from dllm.glm_output import GlmOutput
        parser=GlmOutput()
        self.assertEqual(parser.feed('<|'),[])
        self.assertTrue(parser.pending)
        parser.close()
        parser.close()
        self.assertEqual(parser.pending,'')
        with self.assertRaises(RuntimeError):parser.feed('text')
        with self.assertRaises(RuntimeError):parser.finish(truncated=True)

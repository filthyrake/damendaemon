import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dllm.tokenizer_metadata import (read_tokenizer_config, read_generation_config,
    MAX_TOKENIZER_CONFIG_BYTES, MAX_GENERATION_CONFIG_BYTES)


class TokenizerMetadataTests(unittest.TestCase):
    def test_invalid_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata.json'
            for reader in (read_tokenizer_config, read_generation_config):
                for raw in (b'[]', b'null', b'1', b'"text"', b'\xff',
                            b'{"eos_token_id":1,"eos_token_id":2}',
                            b'{"nested":{"x":1,"x":2}}',
                            b'['*10000+b'0'+b']'*10000):
                    with self.subTest(reader=reader.__name__,raw=raw[:50]):
                        path.write_bytes(raw)
                        with self.assertRaises(ValueError):reader(path)
                path.write_text('{"eos_token_id":[1,2],"added_tokens_decoder":{"0":{"content":"é"}}}',encoding='utf-8')
                self.assertEqual(reader(path)['eos_token_id'],[1,2])
                self.assertEqual(reader(path)['added_tokens_decoder']['0']['content'],'é')

    def test_reads_are_bounded_and_closed_at_real_limits(self):
        for reader, limit in ((read_tokenizer_config,MAX_TOKENIZER_CONFIG_BYTES),
                              (read_generation_config,MAX_GENERATION_CONFIG_BYTES)):
            for extra in (0,1):
                stream=io.BytesIO(b'{}'+b' '*(limit-2+extra))
                with self.subTest(reader=reader.__name__,extra=extra):
                    with patch.object(stream,'read',wraps=stream.read) as read:
                        with patch.object(Path,'open',return_value=stream):
                            if extra:
                                with self.assertRaisesRegex(ValueError,'byte limit'):reader('metadata.json')
                            else:self.assertEqual(reader('metadata.json'),{})
                        read.assert_called_once_with(limit+1)
                    self.assertTrue(stream.closed)

    def test_special_token_text_preserves_strings_and_absent_values(self):
        from dllm.tokenizer_metadata import special_token_text
        self.assertEqual(special_token_text({}),{'bos_token':None,'eos_token':None})
        self.assertEqual(special_token_text({'bos_token':'','eos_token':{'content':'EOS','special':True}}),
                         {'bos_token':'','eos_token':'EOS'})
        for name in ('bos_token','eos_token'):
            for value in ({}, {'content':None}, {'content':[]}, [], True, 4):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    special_token_text({name:value})

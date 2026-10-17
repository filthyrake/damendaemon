import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from damendaemon.tokenizer_metadata import (read_tokenizer_config, read_generation_config,
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
        from damendaemon.tokenizer_metadata import special_token_text
        self.assertEqual(special_token_text({}),{'bos_token':None,'eos_token':None})
        self.assertEqual(special_token_text({'bos_token':'','eos_token':{'content':'EOS','special':True}}),
                         {'bos_token':'','eos_token':'EOS'})
        for name in ('bos_token','eos_token'):
            for value in ({}, {'content':None}, {'content':[]}, [], True, 4, '\ud800', {'content':'\udfff'}):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    special_token_text({name:value})
        self.assertEqual(special_token_text({'bos_token':'世界','eos_token':{'content':'é😀'}}),
                         {'bos_token':'世界','eos_token':'é😀'})

    def test_scalar_template_unicode_and_byte_limit(self):
        from damendaemon.tokenizer_metadata import checkpoint_template
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            for text in ('\ud800','prefix\udfff'):
                with self.subTest(text=ascii(text)),self.assertRaisesRegex(ValueError,'UTF-8'):
                    checkpoint_template(directory,{'chat_template':text})
            with patch('damendaemon.tokenizer_metadata.MAX_CHECKPOINT_TEMPLATE_BYTES',4):
                self.assertEqual(checkpoint_template(directory,{'chat_template':'éé'}),'éé')
                with self.assertRaisesRegex(ValueError,'byte limit'):
                    checkpoint_template(directory,{'chat_template':'ééx'})

    def test_checkpoint_template_precedence_and_bounds(self):
        from damendaemon.tokenizer_metadata import checkpoint_template
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fallback=root/'chat_template.jinja'
            self.assertIsNone(checkpoint_template(root,{}))
            fallback.write_bytes(b'\xff')
            for value in ('','embedded'):
                self.assertEqual(checkpoint_template(root,{'chat_template':value}),value)
            with self.assertRaises(ValueError):checkpoint_template(root,{})
            for value in ([],{},True,1):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    checkpoint_template(root,{'chat_template':value})
            with patch('damendaemon.tokenizer_metadata.MAX_CHECKPOINT_TEMPLATE_BYTES',8):
                for size in (0,8,9):
                    fallback.write_bytes(b'x'*size)
                    if size==9:
                        with self.assertRaisesRegex(ValueError,'byte limit'):checkpoint_template(root,{})
                    else:self.assertEqual(checkpoint_template(root,{'chat_template':None}),'x'*size)
                # The cap is on bytes, including multi-byte UTF-8 text.
                fallback.write_text('é'*5,encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'byte limit'):checkpoint_template(root,{})
            stream=io.BytesIO(b'fallback')
            with patch.object(Path,'exists',return_value=True),patch.object(Path,'open',return_value=stream), \
                    patch.object(stream,'read',wraps=stream.read) as read:
                self.assertEqual(checkpoint_template(root,{}),'fallback')
                from damendaemon.tokenizer_metadata import MAX_CHECKPOINT_TEMPLATE_BYTES
                read.assert_called_once_with(MAX_CHECKPOINT_TEMPLATE_BYTES+1)
            self.assertTrue(stream.closed)

    def test_named_templates_select_only_explicit_default(self):
        from damendaemon.tokenizer_metadata import checkpoint_template
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'chat_template.jinja').write_text('fallback')
            for value in ({'tool_use':'tool','default':'ordinary'},
                          [{'name':'tool_use','template':'tool'},{'name':'default','template':'ordinary'}],
                          [{'name':'default','template':'ordinary'},{'name':'tool_use','template':'tool'}]):
                with self.subTest(value=value):
                    self.assertEqual(checkpoint_template(root,{'chat_template':value}),'ordinary')
                    self.assertEqual(checkpoint_template(root,{'chat_template':value},tools=True),'tool')
            invalid=({'tool_use':'tool'}, [{'name':'other','template':'x'}],
                     [{'name':'default','template':'a'},{'name':'default','template':'b'}],
                     [{'name':'default'}], [{'name':'default','template':'a','extra':1}],
                     {'default':'x','tool_use':None}, {'default':'x','': 'y'},
                     [{'name':[],'template':'x'}], ['default'],
                     {'default':'x',**{str(i):'x' for i in range(64)}})
            for value in invalid:
                with self.subTest(value=value),self.assertRaises(ValueError):
                    checkpoint_template(root,{'chat_template':value})

    def test_named_template_collection_has_utf8_byte_budget(self):
        from damendaemon.tokenizer_metadata import checkpoint_template
        with patch('damendaemon.tokenizer_metadata.MAX_CHECKPOINT_TEMPLATE_BYTES',8):
            self.assertEqual(checkpoint_template('.',{'chat_template':{'default':'é'*4}}),'é'*4)
            for value in ({'default':'é'*5}, {'default':'é'*4,'other':'x'},
                          {'default':'x','other':'\ud800'}):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    checkpoint_template('.',{'chat_template':value})

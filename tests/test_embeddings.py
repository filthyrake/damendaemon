import json
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
import base64
import struct
import torch
from dllm.embeddings import options
from dllm.encoder import config
from dllm.runtime import Runtime


class EmbeddingOutputTests(unittest.TestCase):
    def runtime(self, values, embedding_dimension=None):
        runtime = Runtime.__new__(Runtime)
        self.released = False
        model = SimpleNamespace(task='embedding', config={'hidden_size': 2},
                                encode=lambda inputs: [[1] for _ in inputs])
        if embedding_dimension is not None:model.config['embedding_dimension']=embedding_dimension
        executor = SimpleNamespace(model=model, embed=lambda _: values,
                                   kernel_backend='test-cpu', compute_dtype=torch.float32)
        @contextmanager
        def session(alias):
            try:
                yield executor, 'host', 0
            finally:
                self.released = True
        runtime.session = session
        return runtime

    def test_saved_width_float_and_base64_contract(self):
        values=torch.tensor([[.25],[-.5]])
        for encoding in ('float','base64'):
            result=self.runtime(values,1).embed('model',['first','second'],encoding,dimensions=1)
            actual=[row['embedding'] for row in result['data']]
            if encoding=='base64':actual=[list(struct.unpack('<f',base64.b64decode(row))) for row in actual]
            self.assertEqual(actual,values.tolist())
            with self.assertRaises(ValueError):self.runtime(values,1).embed('model',['first','second'],encoding,dimensions=2)

    def test_invalid_output_never_publishes_and_releases_session(self):
        for values in (torch.tensor([[float('nan'), 0.]]),
                       torch.tensor([[float('inf'), 0.]]),
                       torch.tensor([[-float('inf'), 0.]]),
                       torch.empty((0, 2)), torch.ones((2, 2)),
                       torch.ones((1, 3)), torch.ones(2), torch.ones((1, 1, 2))):
            for encoding in ('float', 'base64'):
                with self.subTest(shape=values.shape, encoding=encoding):
                    runtime = self.runtime(values)
                    with self.assertRaisesRegex(RuntimeError, 'invalid embedding output'):
                        runtime.embed('model', 'hello', encoding)
                    self.assertTrue(self.released)

    def test_valid_float_and_base64_preserve_order_width_and_usage(self):
        values = torch.tensor([[.25, -.5], [0., 1.]])
        for encoding in ('float', 'base64'):
            with self.subTest(encoding=encoding):
                result = self.runtime(values).embed('model', ['first', 'second'], encoding)
                actual = [item['embedding'] for item in result['data']]
                if encoding == 'base64':
                    actual = [list(struct.unpack('<2f', base64.b64decode(row))) for row in actual]
                self.assertEqual(actual, values.tolist())
                self.assertEqual([item['index'] for item in result['data']], [0, 1])
                self.assertEqual(result['usage'], {'prompt_tokens': 2, 'total_tokens': 2})
                self.assertTrue(self.released)


class EmbeddingInputTests(unittest.TestCase):
    def test_text_and_tokens_are_normalized_without_aliasing(self):
        self.assertEqual(options('hello')[0],['hello'])
        ids=[1,2]
        actual=options(ids)[0]
        ids.append(3)
        self.assertEqual(actual,[[1,2]])
        self.assertEqual(options(['a','b'],'base64',384),(['a','b'],'base64',384))

    def test_invalid_encoder_unicode_never_acquires_a_model_session(self):
        from unittest.mock import Mock
        from dllm.reranking import options as rerank_options
        runtime = Runtime.__new__(Runtime)
        runtime.session = Mock(side_effect=AssertionError('must not activate model'))
        for value in ('\ud800', '\udfff', 'prefix\ud800suffix'):
            for call in (lambda: runtime.embed('model', value),
                         lambda: runtime.embed('model', ['valid', value]),
                         lambda: runtime.rerank('model', value, ['valid']),
                         lambda: runtime.rerank('model', 'valid', ['valid', value])):
                with self.subTest(value=ascii(value)), self.assertRaises(ValueError):
                    call()
        runtime.session.assert_not_called()
        text = 'café 世界 😀'
        self.assertEqual(options(text)[0], [text])
        self.assertEqual(rerank_options(text, [text])[:2], (text, [text]))

    def test_invalid_requests_fail_closed(self):
        for value in (None,'',[],[True],[1.5],[-1],[[]],['a',[1]],['x']*65):
            with self.subTest(value=value),self.assertRaises(ValueError):options(value)
        for encoding,dimensions in (('int8',None),('float',True),('float',0),('float',1.5)):
            with self.assertRaises(ValueError):options('text',encoding,dimensions)


class EncoderConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.c={'model_type':'bert','architectures':['BertModel'],'hidden_size':8,'intermediate_size':16,
                'num_attention_heads':2,'num_hidden_layers':2,'max_position_embeddings':512,
                'vocab_size':32,'type_vocab_size':2,'hidden_act':'gelu','pad_token_id':0}
        self.modules=[{'type':'sentence_transformers.models.'+kind,'path':path} for kind,path in
                      (('Transformer',''),('Pooling','1_Pooling'),('Normalize','2_Normalize'))]
        (self.root/'1_Pooling').mkdir()
        (self.root/'1_Pooling/config.json').write_text(json.dumps({'word_embedding_dimension':8,
            'pooling_mode_cls_token':False,'pooling_mode_mean_tokens':True,'pooling_mode_max_tokens':False,
            'pooling_mode_mean_sqrt_len_tokens':False}))
        (self.root/'sentence_bert_config.json').write_text('{"max_seq_length":256}')
        self.write()

    def write(self):
        (self.root/'config.json').write_text(json.dumps(self.c))
        (self.root/'modules.json').write_text(json.dumps(self.modules))

    def test_declared_pipeline_and_context(self):
        parsed=config(self.root)
        self.assertEqual(parsed['task'],'embedding')
        self.assertEqual(parsed['embedding_max_length'],256)

    def test_declared_lowercase_policy_preprocesses_text_only(self):
        from dllm.encoder import EncoderWeights
        from tokenizers import Tokenizer, models, pre_tokenizers
        vocab = {'[UNK]':0, 'HELLO':1, 'hello':2, 'É':3, 'é':4, 'İ':5, 'i̇':6}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        path = self.root/'sentence_bert_config.json'
        for flag in (None, False, True):
            metadata = {'max_seq_length':256}
            if flag is not None:
                metadata['do_lower_case'] = flag
            path.write_text(json.dumps(metadata))
            model = EncoderWeights.__new__(EncoderWeights)
            model.config = config(self.root)
            model.max_context = 256
            model.tokenizer = tokenizer
            with self.subTest(flag=flag):
                self.assertEqual(model.encode(['  HELLO  ', '\tÉ\n', 'İ']),
                                 [[2], [4], [6]] if flag else [[1], [3], [5]])
                # With no whitespace pretokenizer, this requires our strip policy.
                model.tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
                self.assertEqual(model.encode('  HELLO  '), [[2]] if flag else [[1]])
                ids = [1, 3, 5]
                actual = model.encode(ids)
                self.assertEqual(actual, [ids])
                actual[0].append(2)
                self.assertEqual(ids, [1, 3, 5])
        for flag in (None, 0, 1, 'true', [], {}):
            with self.subTest(invalid=flag):
                path.write_text(json.dumps({'max_seq_length':256, 'do_lower_case':flag}))
                with self.assertRaisesRegex(ValueError, 'lowercase'):
                    config(self.root)

    def test_saved_default_prompt_and_truncation_policy(self):
        from dllm.encoder import EncoderWeights
        from tokenizers import Tokenizer, models
        path = self.root/'config_sentence_transformers.json'
        vocabulary = {'[UNK]':0, 'query: hello':1, 'hello':2, 'query:  hello':3}
        model = EncoderWeights.__new__(EncoderWeights)
        model.max_context = 256
        model.tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
        (self.root/'sentence_bert_config.json').write_text('{"max_seq_length":256,"do_lower_case":true}')
        settings = {'prompts':{'query':'QUERY: '}, 'default_prompt_name':'query'}
        path.write_text(json.dumps(settings))
        model.config = config(self.root)
        self.assertEqual(model.encode('HELLO'), [[1]])
        # Prefix before strip preserves the input's initial whitespace internally.
        self.assertEqual(model.encode(' HELLO '), [[3]])
        self.assertEqual(model.encode([2]), [[2]])
        for settings in ({}, {'prompts':{'query':'QUERY: '}, 'default_prompt_name':None},
                         {'prompts':{'query':None}, 'default_prompt_name':'query'}):
            path.write_text(json.dumps(settings))
            model.config = config(self.root)
            self.assertEqual(model.encode('HELLO'), [[2]])
        for settings in ({'truncate_dim':0}, {'truncate_dim':-1}, {'truncate_dim':4.0}, {'truncate_dim':False},
                         {'default_prompt_name':'missing'}, {'default_prompt_name':1},
                         {'prompts':[], 'default_prompt_name':'query'},
                         {'prompts':{'query':True}, 'default_prompt_name':'query'},
                         {'prompts':{'query':'\ud800'}, 'default_prompt_name':'query'}):
            with self.subTest(settings=ascii(settings)):
                path.write_text(json.dumps(settings))
                with self.assertRaises(ValueError):
                    config(self.root)

    def test_compact_mean_pooling_and_published_module_aliases(self):
        expected = config(self.root)
        path = self.root/'1_Pooling/config.json'
        legacy = json.loads(path.read_text())
        aliases = ['sentence_transformers.base.modules.transformer.Transformer',
                   'sentence_transformers.sentence_transformer.modules.pooling.Pooling',
                   'sentence_transformers.base.modules.normalize.Normalize']
        for module, name in zip(self.modules, aliases):
            module['type'] = name
        self.write()
        compact = {'embedding_dimension':8, 'pooling_mode':'mean', 'include_prompt':True}
        for pooling in (legacy, compact, {'embedding_dimension':8, 'pooling_mode':'mean'}):
            with self.subTest(pooling=pooling):
                path.write_text(json.dumps(pooling))
                self.assertEqual(config(self.root), expected)
        for changes in ({'embedding_dimension':8.0}, {'embedding_dimension':True},
                        {'pooling_mode':'cls'}, {'pooling_mode':['mean', 'max']},
                        {'include_prompt':False}, {'include_prompt':1},
                        {'pooling_mode_mean_tokens':True}, {'unknown':False}):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**compact, **changes}))
                with self.assertRaisesRegex(ValueError, 'mean pooling'):
                    config(self.root)
        path.write_text(json.dumps(compact))
        self.modules[1]['type'] = 'untrusted.modules.pooling.Pooling'
        self.write()
        with self.assertRaisesRegex(ValueError, 'pipeline'):
            config(self.root)

    def test_equivalent_serialized_pooling_defaults_are_supported(self):
        # Sentence Transformers v3.4.1 Pooling.config_keys/save includes these
        # defaults even when only masked mean pooling is enabled.
        import itertools
        path=self.root/'1_Pooling/config.json'
        legacy=json.loads(path.read_text());expected=config(self.root)
        optional={'pooling_mode_weightedmean_tokens':False,
                  'pooling_mode_lasttoken':False,'include_prompt':True}
        for enabled in itertools.product((False,True),repeat=3):
            extra={key:value for (key,value),present in zip(optional.items(),enabled) if present}
            with self.subTest(extra=extra):
                path.write_text(json.dumps({**legacy,**extra}))
                self.assertEqual(config(self.root),expected)
        for extra in ({'pooling_mode_weightedmean_tokens':True},
                      {'pooling_mode_lasttoken':True}, {'include_prompt':False},
                      {'pooling_mode_weightedmean_tokens':0},
                      {'pooling_mode_lasttoken':0}, {'include_prompt':1},
                      {'include_prompt':None}, {'unknown':False}):
            with self.subTest(extra=extra):
                path.write_text(json.dumps({**legacy,**extra}))
                with self.assertRaisesRegex(ValueError,'mean pooling'):config(self.root)

    def test_module_metadata_is_bounded_unique_and_well_typed(self):
        valid=json.dumps(self.modules)
        duplicate=valid.replace('"type": "sentence_transformers.models.Pooling"',
            '"type": "unsupported", "type": "sentence_transformers.models.Pooling"')
        for raw in (duplicate, valid+' '*(1024*1024), '['*10000+'0'+']'*10000):
            with self.subTest(raw=raw[:60]):
                (self.root/'modules.json').write_text(raw)
                with self.assertRaises(ValueError):config(self.root)
        for value in (None,True,1,[],{}):
            with self.subTest(path=value):
                self.modules[1]['path']=value;self.write()
                with self.assertRaisesRegex(ValueError,'pooling module path'):config(self.root)

    def test_module_read_has_an_actual_byte_bound_and_closes_source(self):
        import io
        from unittest.mock import Mock,patch
        raw=json.dumps(self.modules).encode()
        stream=io.BytesIO(raw+b' '*(1024*1024-len(raw)))
        stream.read=Mock(wraps=stream.read)
        original=Path.open
        def open_file(path,*args,**kwargs):
            return stream if path==self.root/'modules.json' else original(path,*args,**kwargs)
        with patch.object(Path,'open',new=open_file):
            self.assertEqual(config(self.root)['task'],'embedding')
        stream.read.assert_called_once_with(1024*1024+1)
        self.assertTrue(stream.closed)

    def test_unsupported_architecture_and_pooling(self):
        for change in ({'layer_norm_eps':10**400},{'is_decoder':True},{'position_embedding_type':'relative_key'},
                       {'num_attention_heads':3},{'hidden_act':'relu'},{'layer_norm_eps':float('nan')},
                       {'architectures':['BertForMaskedLM']}):
            with self.subTest(change=change):
                original=dict(self.c);self.c.update(change);self.write()
                with self.assertRaises(ValueError):config(self.root)
                self.c=original
        self.write()
        self.modules[1]['path']='../outside';self.write()
        with self.assertRaises(ValueError):config(self.root)
        self.modules[1]['path']='1_Pooling';self.modules[2]['type']='sentence_transformers.models.Dense';self.write()
        with self.assertRaises(ValueError):config(self.root)

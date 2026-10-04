import json
from pathlib import Path
import tempfile
import unittest
from dllm.embeddings import options
from dllm.encoder import config


class EmbeddingInputTests(unittest.TestCase):
    def test_text_and_tokens_are_normalized_without_aliasing(self):
        self.assertEqual(options('hello')[0],['hello'])
        ids=[1,2]
        actual=options(ids)[0]
        ids.append(3)
        self.assertEqual(actual,[[1,2]])
        self.assertEqual(options(['a','b'],'base64',384),(['a','b'],'base64',384))

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

    def test_unsupported_architecture_and_pooling(self):
        for change in ({'is_decoder':True},{'position_embedding_type':'relative_key'},
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

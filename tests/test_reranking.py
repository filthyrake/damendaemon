import json
from pathlib import Path
import tempfile
import unittest
from dllm.encoder import config
from dllm.reranking import options,ranked,length_groups


class RerankingTests(unittest.TestCase):
    def test_padding_groups_preserve_indices_and_skip_small_savings(self):
        self.assertIsNone(length_groups([[0]*20]*64))
        self.assertIsNone(length_groups([[0]*500]*63+[[0]*512]))
        lengths=[512,24,128,24,256,24,64,24]*8
        groups=length_groups([[0]*n for n in lengths])
        self.assertEqual(sorted(i for group in groups for i in group),list(range(64)))
        self.assertEqual(len(groups),4)
        self.assertEqual(groups[0],list(range(0,64,8)))
        self.assertLessEqual(sum(len(g)*max(lengths[i] for i in g) for g in groups)*2,64*512)
        long_groups=length_groups([[0]*n for n in [8192,512,1024]+[24]*13])
        self.assertEqual(long_groups[:3],[[0],[1],[2]])

    def test_options_and_stable_rank(self):
        docs=['first','second','third']
        q,copy,n,include=options('query',docs,2,True);docs[0]='changed'
        self.assertEqual(copy,['first','second','third'])
        result=ranked([1.,1.,-1.],copy,n,include)
        self.assertEqual([r['index'] for r in result],[0,1])
        self.assertEqual(result[0]['document'],{'text':'first'})
        for scores in ([float('nan')]*3,[float('inf')]*3,[1.]):
            with self.assertRaises(RuntimeError):ranked(scores,copy,2,False)

    def test_malformed_backend_score_shapes_are_runtime_errors(self):
        for scores in (None, 1., '12', [[1.], [2.]], [None, 2.],
                       [True, 2.], ['1', 2.], [10**400, 2.], [1., float('nan')]):
            with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                ranked(scores, ['first', 'second'], 2, False)

    def test_invalid_requests(self):
        for q,docs,n,include in [('', ['doc'],None,False),('q',[],None,False),
                ('q',['doc']*65,None,False),('q',[''],None,False),('q',[{}],None,False),
                ('q',['doc'],0,False),('q',['doc'],2,False),('q',['doc'],True,False),
                ('q',['doc'],None,1)]:
            with self.subTest(values=(q,docs,n,include)),self.assertRaises(ValueError):options(q,docs,n,include)

    def test_single_output_classifier_configuration(self):
        c={'model_type':'bert','architectures':['BertForSequenceClassification'],'hidden_size':8,
            'intermediate_size':16,'num_attention_heads':2,'num_hidden_layers':2,'max_position_embeddings':512,
            'vocab_size':32,'type_vocab_size':2,'hidden_act':'gelu','pad_token_id':0,'id2label':{'0':'LABEL_0'}}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'config.json';path.write_text(json.dumps(c))
            self.assertEqual(config(root)['task'],'rerank')
            self.assertEqual(config(root)['embedding_max_length'],512)
            for changed in ({'num_labels':2},{'problem_type':'single_label_classification'},
                    {'sbert_ce_default_activation_function':'torch.nn.Sigmoid'},{'is_decoder':True}):
                path.write_text(json.dumps({**c,**changed}))
                with self.assertRaises(ValueError):config(root)

    def test_xlm_roberta_position_offset_and_architecture_dispatch(self):
        from dllm.models import config as dispatch
        c={'model_type':'xlm-roberta','architectures':['XLMRobertaForSequenceClassification'],
            'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,'num_hidden_layers':2,
            'max_position_embeddings':8194,'vocab_size':32,'type_vocab_size':1,
            'hidden_act':'gelu','pad_token_id':1,'id2label':{'0':'LABEL_0'}}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'config.json';path.write_text(json.dumps(c))
            self.assertEqual(dispatch(root)['embedding_max_length'],8192)
            self.assertEqual(dispatch(root)['task'],'rerank')
            for change in ({'architectures':['BertForSequenceClassification']},
                    {'architectures':['XLMRobertaForMaskedLM']},{'max_position_embeddings':2}):
                path.write_text(json.dumps({**c,**change}))
                with self.assertRaises(ValueError):dispatch(root)

    def test_roberta_dispatch_and_unsupported_heads(self):
        from dllm.models import config as dispatch
        c={'model_type':'roberta','architectures':['RobertaForSequenceClassification'],
           'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,'num_hidden_layers':2,
           'max_position_embeddings':514,'vocab_size':32,'type_vocab_size':1,
           'hidden_act':'gelu','pad_token_id':1,'num_labels':1}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'config.json';path.write_text(json.dumps(c))
            self.assertEqual(dispatch(root)['task'],'rerank')
            self.assertEqual(dispatch(root)['embedding_max_length'],512)
            for change in ({'num_labels':2},{'architectures':['RobertaForMaskedLM']},
                           {'is_decoder':True},{'max_position_embeddings':2}):
                path.write_text(json.dumps({**c,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):dispatch(root)

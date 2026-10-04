import json
from pathlib import Path
import tempfile
import unittest
from dllm.encoder import config
from dllm.reranking import options,ranked


class RerankingTests(unittest.TestCase):
    def test_options_and_stable_rank(self):
        docs=['first','second','third']
        q,copy,n,include=options('query',docs,2,True);docs[0]='changed'
        self.assertEqual(copy,['first','second','third'])
        result=ranked([1.,1.,-1.],copy,n,include)
        self.assertEqual([r['index'] for r in result],[0,1])
        self.assertEqual(result[0]['document'],{'text':'first'})
        for scores in ([float('nan')]*3,[float('inf')]*3,[1.]):
            with self.assertRaises(RuntimeError):ranked(scores,copy,2,False)

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

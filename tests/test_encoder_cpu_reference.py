"""Tiny independent CPU oracles; no GPU, downloads or model-quality claim."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, models
from transformers import (BertConfig, BertModel, BertForSequenceClassification,
                          XLMRobertaConfig, XLMRobertaForSequenceClassification)
from dllm.encoder import EncoderWeights, EncoderExecutor


class EncoderCPUReferenceTests(unittest.TestCase):
    def check_family(self, family, task, *, compact=False):
        cls = XLMRobertaConfig if family == 'xlm-roberta' else BertConfig
        c = cls(vocab_size=32, hidden_size=8, intermediate_size=16,
                num_hidden_layers=2, num_attention_heads=2,
                max_position_embeddings=32, type_vocab_size=1 if family == 'xlm-roberta' else 2,
                pad_token_id=1 if family == 'xlm-roberta' else 0,
                num_labels=1, problem_type='regression', hidden_dropout_prob=0,
                attention_probs_dropout_prob=0)
        c._attn_implementation = 'eager'
        model_class = (BertModel if task == 'embedding' else
                       XLMRobertaForSequenceClassification if family == 'xlm-roberta' else
                       BertForSequenceClassification)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            reference = model_class(c).eval()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference.save_pretrained(root)
            Tokenizer(models.WordLevel({'[UNK]':0}, unk_token='[UNK]')).save(str(root/'tokenizer.json'))
            if task == 'embedding':
                (root/'1_Pooling').mkdir()
                (root/'modules.json').write_text(json.dumps([
                    {'type':'sentence_transformers.models.'+kind, 'path':path}
                    for kind,path in (('Transformer',''), ('Pooling','1_Pooling'), ('Normalize','2_Normalize'))]))
                (root/'1_Pooling/config.json').write_text(json.dumps({
                    'word_embedding_dimension':8, 'pooling_mode_cls_token':False,
                    'pooling_mode_mean_tokens':True, 'pooling_mode_max_tokens':False,
                    'pooling_mode_mean_sqrt_len_tokens':False}))
                (root/'sentence_bert_config.json').write_text('{"max_seq_length":32}')
                if compact:
                    aliases = ['sentence_transformers.base.modules.transformer.Transformer',
                               'sentence_transformers.sentence_transformer.modules.pooling.Pooling',
                               'sentence_transformers.base.modules.normalize.Normalize']
                    modules = json.loads((root/'modules.json').read_text())
                    for module, alias in zip(modules, aliases):
                        module['type'] = alias
                    (root/'modules.json').write_text(json.dumps(modules))
                    (root/'1_Pooling/config.json').write_text(json.dumps({
                        'embedding_dimension':8, 'pooling_mode':'mean', 'include_prompt':True}))

            model = EncoderWeights(root, 32)
            # Replace device admission only. Actual loading, tensor operations,
            # padding, positions, attention and task heads run unmocked on CPU.
            with patch('dllm.encoder.selected_device', return_value='cpu'), \
                    patch('dllm.encoder.empty_cache'), \
                    patch('dllm.encoder.available_memory', return_value=2**30):
                native = EncoderExecutor(model, use_graph=False)
            self.assertEqual(native.compute_dtype, torch.float32)
            rows = [[2, 5, 6, 7, 3], [2, 8, 3], [2, 9, 10, 11, 12, 3]]
            types = ([[0]*len(row) for row in rows] if family == 'xlm-roberta' else
                     [[0, 0, 1, 1, 1], [0, 0, 1], [0, 0, 0, 1, 1, 1]])
            if task == 'embedding':
                types = [[0]*len(row) for row in rows]
            def compare(tokens, segments):
                length = max(map(len, tokens))
                ids = torch.tensor([row+[c.pad_token_id]*(length-len(row)) for row in tokens])
                mask = torch.tensor([[1]*len(row)+[0]*(length-len(row)) for row in tokens])
                token_types = torch.tensor([row+[0]*(length-len(row)) for row in segments])
                with torch.inference_mode():
                    output = reference(input_ids=ids, attention_mask=mask, token_type_ids=token_types)
                    if task == 'embedding':
                        expected = F.normalize((output.last_hidden_state*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True), dim=-1)
                        actual = native.embed(tokens)
                    else:
                        expected = output.logits.reshape(-1)
                        actual = native.rerank(tokens, segments)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                return actual
            mixed = compare(rows, types)
            singles = torch.cat([compare([row], [segments]) for row,segments in zip(rows, types)])
            torch.testing.assert_close(mixed, singles, atol=1e-6, rtol=1e-5)
            compare(list(reversed(rows)), list(reversed(types)))

    def test_bert_mean_pooling(self):
        self.check_family('bert', 'embedding')

    def test_compact_bert_mean_pooling(self):
        self.check_family('bert', 'embedding', compact=True)

    def test_bert_reranking(self):
        self.check_family('bert', 'rerank')

    def test_xlm_roberta_reranking(self):
        self.check_family('xlm-roberta', 'rerank')

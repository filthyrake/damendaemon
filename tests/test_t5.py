import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from dllm.t5 import config,relative_buckets
from dllm.prompts import validate_lengths,decoder_prefix


class T5Tests(unittest.TestCase):
    def test_penalties_use_decoder_history(self):
        executor=SimpleNamespace(is_encoder_decoder=True,model=SimpleNamespace(config={'decoder_start_token_id':0}))
        self.assertEqual(decoder_prefix(executor,[7,8,1]),[0])
        self.assertEqual(decoder_prefix(SimpleNamespace(),[7,8,1]),[7,8,1])

    def test_attention_width_is_independent_of_model_width(self):
        value={'model_type':'t5','architectures':['T5ForConditionalGeneration'],'is_encoder_decoder':True,
            'd_model':512,'d_ff':1024,'d_kv':64,'num_heads':6,'num_layers':8,'vocab_size':32128,
            'decoder_start_token_id':0,'pad_token_id':0,'eos_token_id':1,'feed_forward_proj':'gated-gelu'}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(value))
            parsed=config(directory)
            self.assertEqual(parsed['d_kv']*parsed['num_heads'],384)
            self.assertEqual(parsed['hidden_size'],512)
            for change in ({'layer_norm_epsilon':10**400},{'num_heads':0},{'relative_attention_num_buckets':3},
                           {'relative_attention_max_distance':8},{'decoder_start_token_id':32128},
                           {'feed_forward_proj':'gated-silu'},{'quantization_config':{'bits':4}}):
                path.write_text(json.dumps({**value,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):config(directory)

    def test_relative_bucket_direction_and_log_boundaries(self):
        positions=torch.arange(3)
        self.assertEqual(relative_buckets(positions,positions).tolist(),[[0,0,0],[1,0,0],[2,1,0]])
        self.assertEqual(relative_buckets(positions,positions,bidirectional=True).tolist(),[[0,17,18],[1,0,17],[2,1,0]])
        distance=torch.tensor([0,15,16,32,64,128,256])
        self.assertEqual(relative_buckets(torch.tensor([256]),256-distance).tolist(),[[0,15,16,21,26,31,31]])

    def test_encoder_and_decoder_limits_are_independent(self):
        encoder=SimpleNamespace(context=512,is_encoder_decoder=True)
        validate_lengths(encoder,511,512)
        for prompt,output in ((513,1),(1,513)):
            with self.assertRaises(ValueError):validate_lengths(encoder,prompt,output)
        with self.assertRaises(ValueError):validate_lengths(SimpleNamespace(context=512),511,2)

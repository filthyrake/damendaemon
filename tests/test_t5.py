import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from damendaemon.t5 import config,relative_buckets
from damendaemon.prompts import validate_lengths,decoder_prefix


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

    def test_saved_generation_tokens_reach_loaded_model(self):
        from transformers import T5Config,T5ForConditionalGeneration,GenerationConfig
        from tokenizers import Tokenizer,models
        from damendaemon.t5 import T5Weights
        from unittest.mock import patch
        with TemporaryDirectory() as directory:
            reference=T5ForConditionalGeneration(T5Config(d_model=8,d_ff=16,d_kv=4,
                num_heads=2,num_layers=1,vocab_size=16,decoder_start_token_id=0,
                pad_token_id=0,eos_token_id=1))
            reference.save_pretrained(directory)
            Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(Path(directory)/'tokenizer.json'))
            saved={'decoder_start_token_id':3,'pad_token_id':4,'eos_token_id':[5,6]}
            generation=Path(directory)/'generation_config.json'
            generation.write_text(json.dumps(saved))
            expected=GenerationConfig.from_pretrained(directory,local_files_only=True)
            loaded=T5Weights(directory,16)
            self.assertEqual(loaded.eos,set(expected.eos_token_id))
            self.assertNotIn(1,loaded.eos)
            self.assertEqual(loaded.config['pad_token_id'],expected.pad_token_id)
            executor=SimpleNamespace(is_encoder_decoder=True,model=loaded)
            self.assertEqual(decoder_prefix(executor,[7,8,1]),[expected.decoder_start_token_id])
            generation.write_text('{}')
            self.assertEqual(config(directory)['decoder_start_token_id'],0)
            self.assertEqual(config(directory)['eos_token_id'],1)
            for bad in ({'eos_token_id':[16]}, {'eos_token_id':True},
                        {'eos_token_id':None}, {'decoder_start_token_id':-1},
                        {'decoder_start_token_id':[3]}, {'pad_token_id':16}):
                generation.write_text(json.dumps(bad))
                with self.subTest(bad=bad),patch('damendaemon.qwen2.weight_files',side_effect=AssertionError('weights opened')):
                    with self.assertRaises(ValueError):T5Weights(directory,16)
            for raw in ('[]','{"eos_token_id":5,"eos_token_id":6}','{"eos_token_id":NaN}'):
                generation.write_text(raw)
                with self.subTest(raw=raw),self.assertRaises(ValueError):config(directory)

    def test_activation_overrides_and_cpu_reference_logits(self):
        from transformers import T5Config,T5ForConditionalGeneration
        from tokenizers import Tokenizer,models
        from damendaemon.t5 import T5Weights,T5Executor
        from unittest.mock import patch
        for projection in ('relu','gated-gelu'):
            with self.subTest(projection=projection),TemporaryDirectory() as directory,torch.random.fork_rng(devices=[]):
                torch.manual_seed(73)
                reference=T5ForConditionalGeneration(T5Config(d_model=8,d_ff=16,d_kv=4,
                    num_heads=2,num_layers=2,vocab_size=16,decoder_start_token_id=0,
                    pad_token_id=0,eos_token_id=1,dropout_rate=0,feed_forward_proj=projection)).eval()
                reference.save_pretrained(directory)
                Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(Path(directory)/'tokenizer.json'))
                loaded=T5Weights(directory,16)
                with patch('damendaemon.t5.selected_device',return_value='cpu'),patch('damendaemon.t5.empty_cache'),patch('damendaemon.t5.available_memory',return_value=2**30):
                    executor=T5Executor(loaded,use_graph=False,max_batch=2,kernels='torch')
                prompts=[[7,8,1],[9,1]]
                actual=executor.prefill_batch(prompts,[1,0])
                for row,prompt in enumerate(prompts):
                    with torch.inference_mode():
                        expected=reference(input_ids=torch.tensor([prompt]),decoder_input_ids=torch.tensor([[0]])).logits[:,-1]
                    torch.testing.assert_close(actual[row:row+1],expected,atol=1e-6,rtol=1e-5)
                actual,_=executor.decode_batch([3,4],[1,0])
                for row,prompt in enumerate(prompts):
                    with torch.inference_mode():
                        expected=reference(input_ids=torch.tensor([prompt]),decoder_input_ids=torch.tensor([[0,3+row]])).logits[:,-1]
                    torch.testing.assert_close(actual[row:row+1],expected,atol=1e-6,rtol=1e-5)
                path=Path(directory)/'config.json';original=json.loads(path.read_text())
                for change in ({'dense_act_fn':'silu'},{'is_gated_act':not original['is_gated_act']},{'is_gated_act':int(original['is_gated_act'])}):
                    path.write_text(json.dumps({**original,**change}))
                    with self.subTest(change=change),patch('damendaemon.qwen2.weight_files',side_effect=AssertionError('weights opened')):
                        with self.assertRaises(ValueError):T5Weights(directory,16)
                overridden=T5Config(**{**original,'dense_act_fn':'silu'})
                self.assertEqual(overridden.dense_act_fn,'silu')
                path.write_text(json.dumps({k:v for k,v in original.items() if k not in ('dense_act_fn','is_gated_act')}))
                self.assertEqual(config(directory)['feed_forward_proj'],projection)

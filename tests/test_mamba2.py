import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from dllm.mamba2 import config


class Mamba2Tests(unittest.TestCase):
    def test_compensated_projection_rejects_lossy_weights_before_gpu_allocation(self):
        from dllm.mamba2 import Mamba2Executor
        for value in (1.0001,1.e5,float('inf'),float('nan')):
            model=SimpleNamespace(quantization='none',config={'hidden_size':768,'intermediate_size':1536},
                weights={'0.in_proj':torch.tensor([[value]])})
            with self.subTest(value=value),patch.dict('os.environ',{'DLLM_MAMBA2_COMPENSATED':'1'}),\
                    patch('dllm.mamba2.selected_device',return_value='cuda'),\
                    self.assertRaisesRegex(ValueError,'exactly representable'):
                Mamba2Executor(model)

    def test_compensated_projection_rejects_unqualified_device_and_quantization(self):
        from dllm.mamba2 import Mamba2Executor
        for device,quantization in (('mps','none'),('cuda','q8_0')):
            with self.subTest(device=device,quantization=quantization),\
                    patch.dict('os.environ',{'DLLM_MAMBA2_COMPENSATED':'1'}),\
                    patch('dllm.mamba2.selected_device',return_value=device),\
                    self.assertRaisesRegex(ValueError,'CUDA and unquantized'):
                Mamba2Executor(SimpleNamespace(quantization=quantization))

    def test_reject_unqualified_groups_geometry_and_precision(self):
        value={'model_type':'mamba2','architectures':['Mamba2ForCausalLM'],'hidden_size':768,
            'num_heads':24,'head_dim':64,'state_size':128,'conv_kernel':4,'vocab_size':50288,
            'num_hidden_layers':24,'n_groups':1,'expand':2,'hidden_act':'silu','rms_norm':True,
            'residual_in_fp32':True,'use_bias':False,'use_conv_bias':True,'tie_word_embeddings':True,
            'layer_norm_epsilon':1.e-5,'eos_token_id':0}
        with TemporaryDirectory() as directory:
            file=Path(directory)/'config.json';file.write_text(json.dumps(value))
            parsed=config(directory)
            self.assertEqual(parsed['conv_dim'],1792)
            self.assertEqual(parsed['intermediate_size'],1536)
            from dllm.models import config as dispatched_config
            file.write_text(json.dumps({**value,'time_step_limit':[0.0,float('inf')]}))
            for reader in (config,dispatched_config):
                with self.subTest(reader=reader.__module__):
                    explicit=reader(directory)
                    self.assertEqual(explicit['time_step_limit'],[0.0,float('inf')])
                    self.assertEqual(explicit['intermediate_size'],1536)
            for change in ({'layer_norm_epsilon':10**400},{'n_groups':2},{'n_groups':True},{'head_dim':63},{'state_size':257},
                    {'conv_kernel':17},{'time_step_limit':[.01,1]},{'hidden_act':'relu'},
                    {'residual_in_fp32':False},{'use_bias':1},{'eos_token_id':50288},
                    {'quantization_config':{'bits':8}},{'layer_norm_epsilon':float('nan')}):
                file.write_text(json.dumps({**value,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):config(directory)

    def test_saved_eos_replaces_default_before_weights(self):
        from transformers import Mamba2Config,Mamba2ForCausalLM,GenerationConfig
        from tokenizers import Tokenizer,models
        from dllm.mamba2 import Mamba2Weights
        with TemporaryDirectory() as directory:
            reference=Mamba2ForCausalLM(Mamba2Config(hidden_size=8,state_size=4,
                num_hidden_layers=1,expand=2,conv_kernel=4,num_heads=4,head_dim=4,
                n_groups=1,vocab_size=16,eos_token_id=1))
            reference.save_pretrained(directory)
            Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(Path(directory)/'tokenizer.json'))
            generation=Path(directory)/'generation_config.json'
            generation.write_text(json.dumps({'eos_token_id':[5,6]}))
            expected=GenerationConfig.from_pretrained(directory,local_files_only=True)
            self.assertEqual(Mamba2Weights(directory,16).eos,set(expected.eos_token_id))
            generation.write_text('{}')
            self.assertEqual(Mamba2Weights(directory,16).eos,{1})
            for bad in (None,True,[],[16],-1,'5'):
                generation.write_text(json.dumps({'eos_token_id':bad}))
                with self.subTest(bad=bad),patch('dllm.qwen2.weight_files',side_effect=AssertionError('weights opened')):
                    with self.assertRaises(ValueError):Mamba2Weights(directory,16)

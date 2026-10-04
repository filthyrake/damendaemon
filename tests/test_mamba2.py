import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dllm.mamba2 import config


class Mamba2Tests(unittest.TestCase):
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
            for change in ({'n_groups':2},{'n_groups':True},{'head_dim':63},{'state_size':257},
                    {'conv_kernel':17},{'time_step_limit':[.01,1]},{'hidden_act':'relu'},
                    {'residual_in_fp32':False},{'use_bias':1},{'eos_token_id':50288},
                    {'quantization_config':{'bits':8}},{'layer_norm_epsilon':float('nan')}):
                file.write_text(json.dumps({**value,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):config(directory)

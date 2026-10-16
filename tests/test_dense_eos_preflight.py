"""Exercise real constructor metadata checks without allocating model weights."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from tokenizers import Tokenizer, models
from dllm.qwen2 import ModelWeights


class DenseEosPreflightTests(unittest.TestCase):
    def test_merged_eos_ids_are_bounded_before_weight_cache(self):
        cases = [(3,None,False),([1,3],None,False),(1,3,False),
                 (None,[1,3],False),(1,10**100,False),(None,None,False),
                 (1,None,True),(None,2,True),(1,[1,2],True)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            (root/'tokenizer_config.json').write_text('{}')
            Tokenizer(models.WordLevel({'unknown':0,'one':1,'two':2},unk_token='unknown')).save(str(root/'tokenizer.json'))
            for config_eos,generation_eos,valid in cases:
                with self.subTest(config_eos=config_eos,generation_eos=generation_eos),ExitStack() as stack:
                    generation=root/'generation_config.json';generation.unlink(missing_ok=True)
                    if generation_eos is not None:generation.write_text(json.dumps({'eos_token_id':generation_eos}))
                    config={'model_type':'qwen2','max_position_embeddings':32,'vocab_size':3,'eos_token_id':config_eos}
                    stack.enter_context(patch('dllm.qwen2.decoder_config',return_value=config))
                    stack.enter_context(patch('dllm.qwen2.selected_device',return_value='cpu'))
                    stack.enter_context(patch('dllm.qwen2.weight_files',return_value=[]))
                    stack.enter_context(patch('dllm.layer_stream.enabled',return_value=False))
                    cache=stack.enter_context(patch('dllm.weight_cache.configured_cache',side_effect=RuntimeError('cache boundary reached')))
                    if valid:
                        with self.assertRaisesRegex(RuntimeError,'cache boundary reached'):ModelWeights(root,16,'none')
                        cache.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):ModelWeights(root,16,'none')
                        cache.assert_not_called()

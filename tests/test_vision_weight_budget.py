import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch
from dllm.qwen2 import weight_bytes


class VisionBudgetTests(unittest.TestCase):
    def test_quantized_text_keeps_separate_vision_storage_and_ignores_mtp(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'config.json').write_text(json.dumps({}))
            save_file({
                'model.language_model.embed_tokens.weight':torch.zeros(64,32),
                'model.language_model.norm.weight':torch.zeros(32),
                'model.language_model.layers.0.self_attn.q_proj.weight':torch.zeros(32,32),
                'model.visual.patch_embed.proj.weight':torch.zeros(8,3,2,16,16),
                'model.visual.pos_embed.weight':torch.zeros(2,8),
                'model.visual.blocks.0.norm1.weight':torch.zeros(8),
                'mtp.unused.weight':torch.zeros(32,32),
            },str(path/'model.safetensors'))
            config={'model_type':'qwen3_5_text','tie_word_embeddings':True,'hidden_size':1024,'num_hidden_layers':24}
            with patch('dllm.qwen2.decoder_config',return_value=config):
                with patch.dict(os.environ,{'DLLM_EXPERIMENTAL_VISION':'0'}):
                    self.assertEqual(weight_bytes(path,quantization='none'),6272)
                    self.assertEqual(weight_bytes(path,quantization='q8_0'),7488)
                # 24,576 bytes patch projections, 64 positions, 32 normalization.
                with patch.dict(os.environ,{'DLLM_EXPERIMENTAL_VISION':'1'}):
                    self.assertEqual(weight_bytes(path,quantization='none'),6272+24672)
                    self.assertEqual(weight_bytes(path,quantization='q8_0'),7488+24672)


if __name__=='__main__':unittest.main()

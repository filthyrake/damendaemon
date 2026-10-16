"""Actual recurrent checkpoints and independent CPU logits; no GPU qualification."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models
from transformers import MambaConfig, MambaForCausalLM, Mamba2Config, Mamba2ForCausalLM
from dllm.mamba import MambaWeights, MambaExecutor
from dllm.mamba2 import Mamba2Weights, Mamba2Executor


class MambaCPUReferenceTests(unittest.TestCase):
    def check_family(self, version, bias):
        config_type, reference_type, weights_type, executor_type = (
            (MambaConfig, MambaForCausalLM, MambaWeights, MambaExecutor) if version == 1 else
            (Mamba2Config, Mamba2ForCausalLM, Mamba2Weights, Mamba2Executor))
        c = config_type(hidden_size=8, state_size=4, num_hidden_layers=2, expand=2,
                        conv_kernel=4, time_step_rank=2, vocab_size=16, eos_token_id=1,
                        use_bias=bias, use_conv_bias=bias,
                        **({'num_heads':4, 'head_dim':4, 'n_groups':1, 'chunk_size':4}
                           if version == 2 else {}))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            reference = reference_type(c).eval()
        with tempfile.TemporaryDirectory() as folder:
            reference.save_pretrained(folder)
            Tokenizer(models.WordLevel({'[UNK]':0}, unk_token='[UNK]')).save(str(Path(folder)/'tokenizer.json'))
            model = weights_type(folder, 32)
            # Replace only device admission; loading and recurrence are real.
            with ExitStack() as stack:
                for module in ('dllm.mamba', 'dllm.mamba2'):
                    stack.enter_context(patch(module+'.selected_device', return_value='cpu'))
                stack.enter_context(patch('dllm.mamba.empty_cache'))
                stack.enter_context(patch('dllm.mamba.available_memory', return_value=2**30))
                stack.enter_context(patch.dict('os.environ', {
                    'DLLM_MAMBA_FUSED_DT':'0', 'DLLM_MAMBA2_COMPENSATED':'0', 'DLLM_MAMBA2_SSD':'0'}))
                native = executor_type(model, use_graph=False, kernels='torch')
            self.assertEqual(native.device, 'cpu')
            self.assertEqual(native.kernel_backend, 'torch')

            def expected(prompts):
                # Fresh full-prefix evaluations are independent of native slot
                # state, chunk partitioning and continuation cache management.
                return torch.cat([reference(torch.tensor([prompt]), use_cache=False).logits[:, -1]
                                  for prompt in prompts])

            with torch.inference_mode():
                prompts = [[2,3,4,5,6], [2,8], [4,3,2,6,7]]
                slots = [2,0,3]
                actual = native.prefill_batch(prompts, slots)
                torch.testing.assert_close(actual, expected(prompts), atol=1e-6, rtol=1e-5)
                histories = dict(zip(slots, prompts))
                for assigned, tokens in (([3,2,0], [3,4,5]), ([0,3], [6,7]), ([2,0,3], [8,9,10])):
                    for slot, token in zip(assigned, tokens):
                        histories[slot] = histories[slot] + [token]
                    actual, choices = native.decode_batch(tokens, assigned)
                    target = expected([histories[slot] for slot in assigned])
                    torch.testing.assert_close(actual, target, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(choices, actual.argmax(-1))
                # Continue across and beyond the four-token convolution window.
                prompt = [2,3,4,5,6,7,8,9,10]
                for cuts in ((1,3), (2,6), (5,8)):
                    first, second = cuts
                    self.assertIsNone(native.prefill_chunk(prompt[:first], 1, 0, final=False))
                    self.assertIsNone(native.prefill_chunk(prompt[first:second], 1, first, final=False))
                    actual = native.prefill_chunk(prompt[second:], 1, second)
                    torch.testing.assert_close(actual, expected([prompt]), atol=1e-6, rtol=1e-5)
                # Reset a previously used slot with a prompt shorter than the
                # convolution window, then compare its continuation from scratch.
                actual = native.prefill_batch([[11]], [3])
                torch.testing.assert_close(actual, expected([[11]]), atol=1e-6, rtol=1e-5)
                actual, _ = native.decode_batch([12], [3])
                torch.testing.assert_close(actual, expected([[11,12]]), atol=1e-6, rtol=1e-5)

    def test_mamba1_checkpoint_and_recurrent_state(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                self.check_family(1, bias)

    def test_mamba2_checkpoint_and_recurrent_state(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                self.check_family(2, bias)

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from dllm.distributed_cli import prompt_tokens


class DistributedCliTests(unittest.TestCase):
    def args(self, **updates):
        values = dict(prompt_tokens_file=None, prompt=None, raw=False, max_tokens=2, context=8)
        return SimpleNamespace(**(values | updates))

    def model(self):
        return SimpleNamespace(task='generation', raw_tokens=Mock(return_value=[1, 2]),
                               chat_tokens=Mock(return_value=[3, 4, 5]))

    def test_chat_and_raw_use_shared_prompt_dispatch(self):
        model = self.model()
        self.assertEqual(prompt_tokens(self.args(prompt='hello'), model), [3, 4, 5])
        model.chat_tokens.assert_called_once_with([{'role':'user','content':'hello'}])
        self.assertEqual(prompt_tokens(self.args(prompt='raw', raw=True), model), [1, 2])
        model.raw_tokens.assert_called_once_with('raw')

    def test_token_file_is_bounded_and_must_be_a_list(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tokens.json'
            args = self.args(prompt_tokens_file=str(path))
            path.write_text(json.dumps([1, 2]))
            self.assertEqual(prompt_tokens(args, model), [1, 2])
            model.raw_tokens.assert_called_with([1, 2])
            path.write_text('"text"')
            with self.assertRaisesRegex(ValueError, 'JSON token list'):prompt_tokens(args, model)
            path.write_bytes(b' ' * 32769)
            with self.assertRaisesRegex(ValueError, '32 KiB'):prompt_tokens(args, model)

    def test_missing_and_over_budget_input_rejected(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'coordinator requires'):prompt_tokens(self.args(), model)
        with self.assertRaisesRegex(ValueError, 'exceeds distributed context'):
            prompt_tokens(self.args(prompt='hello', max_tokens=6), model)
        with self.assertRaisesRegex(ValueError, '64 KiB'):
            prompt_tokens(self.args(prompt='x' * 65537), model)

    def test_main_cli_delegates_without_constructing_local_runtime(self):
        from dllm.cli import main
        with patch('sys.argv', ['dllm', 'distributed-run', '/model', '--prompt', 'hello']), \
                patch('dllm.distributed_cli.main') as distributed:
            main()
        distributed.assert_called_once_with(['/model', '--prompt', 'hello'])

    def test_explicit_gpu_selection_required_before_cuda_access(self):
        from dllm.distributed import StagePair
        with patch.dict('os.environ', {'WORLD_SIZE':'2','RANK':'0','LOCAL_RANK':'0','CUDA_VISIBLE_DEVICES':'0'}, clear=True), \
                patch('dllm.distributed.dist.is_initialized', return_value=False), \
                patch('dllm.distributed.torch.cuda.is_available') as available:
            with self.assertRaisesRegex(ValueError, 'explicit distinct GPU UUIDs'):
                StagePair('/unused', 512, 'none', lambda model: [1])
            available.assert_not_called()

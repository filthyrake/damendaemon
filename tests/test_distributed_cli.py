import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from dllm.distributed_cli import prompt_tokens, prepare_requests


class DistributedCliTests(unittest.TestCase):
    def args(self, **updates):
        values = dict(prompt_tokens_file=None, requests_file=None, prompt=None, raw=False,
                      ignore_eos=False, max_tokens=2, context=8)
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

    def test_session_prepares_all_requests_and_individual_budgets(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'requests.json'
            path.write_text(json.dumps([{'prompt':'chat'}, {'tokens':[1, 2], 'max_tokens':3, 'ignore_eos':True}]))
            self.assertEqual(prepare_requests(self.args(requests_file=str(path)), model), [
                dict(tokens=[3, 4, 5], max_tokens=2, ignore_eos=False),
                dict(tokens=[1, 2], max_tokens=3, ignore_eos=True)])

    def test_session_rejects_malformed_late_request_and_unbounded_inputs(self):
        invalid = [[], [{}], [{'prompt':'ok', 'tokens':[1]}], [{'prompt':'ok', 'unknown':1}],
                   [{'tokens':'text'}], [{'tokens':[1], 'raw':True}], [{'prompt':42}],
                   [{'prompt':'ok', 'max_tokens':True}], [{'prompt':'ok', 'max_tokens':6}],
                   [{'prompt':'ok', 'raw':1}], [{'prompt':'ok', 'ignore_eos':'yes'}],
                   [{'prompt':'ok'}, {'prompt':'x' * 65537}], [{'prompt':'ok'}] * 65]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'requests.json'
            args = self.args(requests_file=str(path))
            for rows in invalid:
                with self.subTest(rows=str(rows)[:80]):
                    path.write_text(json.dumps(rows))
                    with self.assertRaises(ValueError):prepare_requests(args, self.model())
            path.write_bytes(b' ' * 1048577)
            with self.assertRaisesRegex(ValueError, '1 MiB'):prepare_requests(args, self.model())

    def test_release_completes_before_slot_reuse(self):
        from dllm.distributed import StagePair
        from dllm.stage_protocol import StageState, Command, PREFILL, RELEASE
        pair = StagePair.__new__(StagePair)
        pair.rank = 0
        pair.state = StageState(1, 512, 1)
        pair.state.commit(pair.state.begin(Command(1, 1, PREFILL, 31, ((0, 1, 0),)).encode()))
        pair.channel = Mock()
        pair.channel.begin.side_effect = lambda command: pair.state.begin(command.encode())
        pair.channel.finish.side_effect = pair.state.commit
        pair.release()
        self.assertEqual(pair.channel.begin.call_args.args[0].operation, RELEASE)
        self.assertEqual(pair.state.lengths, [0])
        self.assertEqual(pair.state.generations, [1])
        pair.release()
        pair.channel.begin.assert_called_once()
        self.assertEqual(pair.state.begin(Command(1, 3, PREFILL, 255, ((0, 2, 0),)).encode()).entries,
                         ((0, 2, 0),))

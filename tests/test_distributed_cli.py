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

    def test_distributed_precision_guard_is_independent_of_local_partition_support(self):
        from dllm.distributed import StagePair
        with patch('dllm.distributed.dist.is_initialized', return_value=False), \
                patch('dllm.distributed.torch.cuda.is_available') as available:
            with self.assertRaisesRegex(ValueError, 'FP16, Q8 or Q4'):
                StagePair('/unused', 512, 'awq', lambda model: [])
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
        pair.max_batch = 1
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

    def test_invalid_batch_capacity_rejects_before_cuda_or_rendezvous(self):
        from dllm.distributed import StagePair
        with patch('dllm.distributed.torch.cuda.is_available') as available:
            for capacity in (0,5,True,1.5):
                with self.subTest(capacity=capacity),self.assertRaisesRegex(ValueError,'batch capacity'):
                    StagePair('/unused',512,'q8_0',lambda model:[],max_batch=capacity)
            available.assert_not_called()

    def test_explicit_slots_preserve_order_and_generation_across_reuse(self):
        from dllm.distributed import StagePair
        from dllm.stage_protocol import StageState
        pair=StagePair.__new__(StagePair);pair.rank=0;pair.max_batch=4
        pair.model=SimpleNamespace(config={'vocab_size':128})
        pair.state=StageState(1,32,4)
        commands=[]
        def exchange(command,tokens):
            commands.append(command)
            pair.state.commit(pair.state.begin(command.encode()))
            return [65]*len(command.entries)
        pair._exchange=exchange
        pair.channel=Mock()
        pair.channel.begin.side_effect=lambda command:pair.state.begin(command.encode())
        pair.channel.finish.side_effect=pair.state.commit
        pair.prefill([1,2],3);pair.prefill([3],1)
        self.assertEqual(pair.decode([65,65],[3,1]),[65,65])
        self.assertEqual(commands[-1].entries,((3,1,2),(1,1,1)))
        with self.assertRaisesRegex(ValueError,'occupied'):pair.prefill([4],3)
        pair.release([3]);pair.prefill([4],3)
        self.assertEqual(commands[-1].entries,((3,2,0),))
        self.assertEqual(pair.state.lengths,[0,2,0,1])
        for slots in ([1,1],[True],[4],[]):
            with self.subTest(slots=slots),self.assertRaises(ValueError):pair.decode([65]*len(slots),slots)

    def test_output_failure_payload_is_not_a_generated_token(self):
        import torch
        from dllm.distributed import StagePair
        pair=StagePair.__new__(StagePair);pair.model=SimpleNamespace(config={'vocab_size':128})
        self.assertEqual(pair._result_tokens(torch.tensor([0,65,127])),[0,65,127])
        with self.assertRaisesRegex(RuntimeError,'output stage failed'):
            pair._result_tokens(torch.tensor([-1,-1]))
        for values in ([128],[-2],[65,128]):
            with self.assertRaisesRegex(RuntimeError,'invalid distributed output'):
                pair._result_tokens(torch.tensor(values))

    def test_output_exception_uses_existing_result_transfer_before_raising(self):
        import torch
        from dllm.distributed import StagePair
        from dllm.stage_protocol import Command, DECODE
        pair=StagePair.__new__(StagePair);pair.stage=Mock();pair.channel=Mock()
        failure=RuntimeError('injected stage error')
        pair.stage.run_decode.side_effect=failure
        command=Command(1,3,DECODE,2,((0,1,4),(1,1,8)))
        payload=torch.tensor([-1,-1],dtype=torch.int64)
        pair.failure_tokens=payload
        with patch('dllm.distributed.torch.full',return_value=payload) as allocate:
            with self.assertRaisesRegex(RuntimeError,'injected stage error'):
                pair._output_tokens(command,None,None,None)
        allocate.assert_not_called()
        pair.channel.complete_tokens.assert_called_once()
        sent_command,sent_payload=pair.channel.complete_tokens.call_args.args
        self.assertEqual(sent_command,command)
        self.assertTrue(torch.equal(sent_payload,payload))

    def test_successful_output_selection_does_not_add_a_transfer(self):
        import torch
        from dllm.distributed import StagePair
        from dllm.stage_protocol import Command, DECODE
        pair=StagePair.__new__(StagePair);pair.stage=Mock();pair.channel=Mock()
        pair.stage.run_decode.return_value=torch.tensor([[1.,3.,2.],[4.,2.,1.]])
        command=Command(1,3,DECODE,2,((0,1,4),(1,1,8)))
        result=pair._output_tokens(command,None,None,None)
        self.assertEqual(result.tolist(),[1,0])
        pair.channel.complete_tokens.assert_not_called()

    def test_packed_output_failure_replies_once_per_prompt_not_activation_row(self):
        import torch
        from dllm.distributed import StagePair
        from dllm.stage_protocol import Command, PREFILL_BATCH
        pair=StagePair.__new__(StagePair);pair.stage=Mock();pair.channel=Mock()
        pair.stage.run_prefill_packed.side_effect=RuntimeError('packed stage error')
        pair.failure_tokens=torch.full((4,),-1,dtype=torch.int64)
        command=Command(1,1,PREFILL_BATCH,11,((3,1,4),(0,1,7)))
        with self.assertRaisesRegex(RuntimeError,'packed stage error'):
            pair._output_tokens(command,None,None,None)
        pair.stage.run_prefill_packed.assert_called_once_with(None,[4,7],[3,0])
        pair.channel.complete_tokens.assert_called_once()
        sent_command,payload=pair.channel.complete_tokens.call_args.args
        self.assertEqual(sent_command,command)
        self.assertEqual(payload.tolist(),[-1,-1])

    def test_packed_prompt_metadata_is_bounded_and_result_order_is_preserved(self):
        from dllm.distributed import StagePair
        from dllm.stage_protocol import StageState, PREFILL_BATCH
        pair=StagePair.__new__(StagePair);pair.rank=0;pair.max_batch=4;pair.packed_prefill=True
        pair.model=SimpleNamespace(config={'vocab_size':128});pair.state=StageState(1,8,4)
        pair._exchange=Mock(return_value=[65,66])
        self.assertEqual(pair.prefill_batch([[1,2],[3,4,5]],[3,0]),[65,66])
        command,tokens=pair._exchange.call_args.args
        self.assertEqual(command.operation,PREFILL_BATCH)
        self.assertEqual(command.entries,((3,1,2),(0,1,3)))
        self.assertEqual(command.rows,5);self.assertEqual(tokens,[1,2,3,4,5])
        pair._exchange.reset_mock()
        for prompts,slots in (([[1]*5,[2]*4],[0,1]),([[1],[2]],[0,0]),([[1],[128]],[0,1])):
            with self.assertRaises(ValueError):pair.prefill_batch(prompts,slots)
        pair._exchange.assert_not_called()
        pair.state.lengths[0]=1
        with self.assertRaises(ValueError):pair.prefill_batch([[1],[2]],[0,1])
        pair._exchange.assert_not_called()

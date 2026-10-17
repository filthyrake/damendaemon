from types import SimpleNamespace
import unittest
from unittest.mock import patch

from damendaemon.token_observation import TokenObservation
from damendaemon.scheduler import BatchedRuntime, Pending


class TokenObservationTests(unittest.TestCase):
    def test_empty_single_and_sequential_observations(self):
        timing = TokenObservation()
        self.assertIsNone(timing.report()['first_sample_seconds'])
        timing.observe(2)
        self.assertIsNone(timing.report()['max_gap_seconds'])
        timing.observe(2.25)
        timing.observe(3)
        result = timing.report()
        self.assertEqual(result['sampled_tokens'], 3)
        self.assertEqual(result['first_sample_seconds'], 2)
        self.assertEqual(result['last_sample_seconds'], 3)
        self.assertEqual(result['mean_gap_seconds'], .5)
        self.assertEqual(result['max_gap_seconds'], .75)
        self.assertEqual(result['interval_tokens_per_second'], 2)
        self.assertEqual(len(vars(timing)), 5)

    def test_speculative_burst_permanently_invalidates_token_gaps(self):
        timing = TokenObservation()
        timing.observe(1)
        timing.observe(2, sequential=False)
        timing.observe(2)
        timing.observe(4)
        result = timing.report()
        self.assertFalse(result['sequential_token_timing'])
        self.assertEqual(result['sampled_tokens'], 4)
        self.assertEqual(result['last_sample_seconds'], 4)
        for key in ('mean_gap_seconds', 'max_gap_seconds', 'interval_tokens_per_second'):
            self.assertIsNone(result[key])

    def test_zero_interval_does_not_invent_rate(self):
        timing = TokenObservation()
        timing.observe(1); timing.observe(1)
        self.assertEqual(timing.report()['max_gap_seconds'], 0)
        self.assertIsNone(timing.report()['interval_tokens_per_second'])

    def test_scheduler_counts_empty_text_and_eos_before_final_formatting(self):
        runtime = object.__new__(BatchedRuntime)
        request = Pending('a', [], 3, start=10)
        executor = SimpleNamespace(model=SimpleNamespace(eos={9}, quantization='none'),
                                   kernel_backend='fixture', residual_dtype='float32')
        with patch.object(runtime, '_finish'), patch('damendaemon.scheduler.decode_output', return_value=''), \
                patch('damendaemon.scheduler.time.perf_counter', side_effect=[11, 13, 20]):
            self.assertFalse(runtime._emit(request, 1, executor, 5))
            self.assertTrue(runtime._emit(request, 9, executor, 5))
        result = request.final['timing']['token_observation']
        self.assertEqual(result['sampled_tokens'], 2)
        self.assertEqual(result['last_sample_seconds'], 3)
        self.assertEqual(result['interval_tokens_per_second'], .5)
        self.assertEqual(request.final['timing']['total_seconds'], 10)
        self.assertEqual(request.final['usage']['completion_tokens'], 2)

    def test_scheduler_speculative_final_has_no_sequential_rate(self):
        runtime = object.__new__(BatchedRuntime)
        request = Pending('a', [], 2, start=10)
        executor = SimpleNamespace(model=SimpleNamespace(eos=set(), quantization='none'),
                                   kernel_backend='fixture', residual_dtype='float32')
        with patch.object(runtime, '_finish'), patch('damendaemon.scheduler.decode_output', return_value=''), \
                patch('damendaemon.scheduler.time.perf_counter', side_effect=[11, 13, 14]):
            runtime._emit(request, 1, executor, 5)
            request.speculative_blocks = 1
            runtime._emit(request, 2, executor, 5)
        self.assertFalse(request.final['timing']['token_observation']['sequential_token_timing'])
        self.assertIsNone(request.final['timing']['token_observation']['interval_tokens_per_second'])

    def test_ordinary_runtime_includes_consumer_pause_but_excludes_final_resume(self):
        from contextlib import nullcontext
        from unittest.mock import Mock
        from damendaemon.runtime import Runtime
        runtime = object.__new__(Runtime)
        runtime.context = 4  # Admission preflight reads the configured context.
        model = SimpleNamespace(eos={9}, quantization='none', config={'model_type': 'qwen3'},
                                tokenizer=SimpleNamespace(decode=lambda *a, **k: ''))
        executor = SimpleNamespace(model=model, kernel_backend='fixture', residual_dtype='float32',
                                   prefill=Mock(), decode=lambda token: (None, SimpleNamespace(item=lambda: 9)))
        runtime.session = lambda alias: nullcontext((executor, 'fixture', 0))
        sampler = Mock(constraint=None, has_adjustments=False)
        sampler.select.return_value.item.return_value = 1
        now = [10]
        with patch('damendaemon.sampling.Sampler', return_value=sampler), \
                patch('damendaemon.prompts.encode', return_value=[1]), patch('damendaemon.prompts.validate_lengths'), \
                patch('damendaemon.runtime.time.perf_counter', side_effect=lambda: now[0]):
            stream = runtime.generate('a', None, 2, raw_prompt=[1])
            self.assertEqual(next(stream)['token'], 1)
            now[0] = 14  # Consumer resumes four seconds later.
            self.assertEqual(next(stream)['token'], 9)
            now[0] = 30  # Final resume must not extend sample interval.
            final = next(stream)
            stream.close()
        observation = final['timing']['token_observation']
        self.assertEqual(observation['mean_gap_seconds'], 4)
        self.assertEqual(observation['interval_tokens_per_second'], .25)
        self.assertEqual(observation['last_sample_seconds'], 4)
        self.assertEqual(final['timing']['total_seconds'], 20)

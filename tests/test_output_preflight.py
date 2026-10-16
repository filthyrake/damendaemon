"""Impossible output requests cannot cause model activation or queue admission."""
import unittest
from unittest.mock import Mock
from dllm.runtime import Runtime
from dllm.scheduler import BatchedRuntime


class OutputPreflightTests(unittest.TestCase):
    def test_invalid_generation_flags_before_ownership(self):
        direct=Runtime.__new__(Runtime);direct.context=16
        direct.session=Mock(side_effect=AssertionError('session acquired'))
        scheduled=BatchedRuntime.__new__(BatchedRuntime);scheduled.context=16
        scheduled.models={'a':'unused'};scheduled.admission=Mock()
        for invalid in (0,1,'false',[],{}):
            for runtime in (direct,scheduled):
                with self.subTest(thinking=invalid,runtime=type(runtime).__name__),self.assertRaisesRegex(ValueError,'boolean'):
                    list(runtime.generate('a',[],max_tokens=1,enable_thinking=invalid))
            with self.subTest(ignore_eos=invalid),self.assertRaisesRegex(ValueError,'boolean'):
                list(direct.generate('a',[],max_tokens=1,ignore_eos=invalid))
        direct.session.assert_not_called();scheduled.admission.acquire.assert_not_called()

    def test_reasoning_effort_syntax_before_ownership(self):
        direct=Runtime.__new__(Runtime);direct.context=16
        direct.session=Mock(side_effect=RuntimeError('session boundary'))
        scheduled=BatchedRuntime.__new__(BatchedRuntime);scheduled.context=16
        scheduled.models={'a':'unused'};scheduled.admission=Mock()
        scheduled.admission.acquire.side_effect=RuntimeError('admission boundary')
        for value in (True,1,'','extreme',[],{}):
            for runtime in (direct,scheduled):
                with self.subTest(value=value,runtime=type(runtime).__name__),self.assertRaisesRegex(ValueError,'reasoning_effort'):
                    list(runtime.generate('a',[],max_tokens=1,reasoning_effort=value))
        direct.session.assert_not_called();scheduled.admission.acquire.assert_not_called()
        for value in ('low','medium','high'):
            for runtime,boundary in ((direct,'session boundary'),(scheduled,'admission boundary')):
                with self.subTest(value=value,runtime=type(runtime).__name__),self.assertRaisesRegex(RuntimeError,boundary):
                    list(runtime.generate('a',[],max_tokens=1,reasoning_effort=value))

    def test_direct_raw_chat_conflicts_before_session(self):
        runtime=Runtime.__new__(Runtime);runtime.context=16
        runtime.session=Mock(side_effect=RuntimeError('session boundary'))
        for options in ({'messages':[]},{'messages':[{'role':'user','content':'x'}]},
                        {'messages':None,'enable_thinking':False},
                        {'messages':None,'enable_thinking':True},
                        {'messages':None,'reasoning_effort':'low'}):
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,'raw prompts cannot'):
                list(runtime.generate('a',raw_prompt=[1],max_tokens=1,**options))
        runtime.session.assert_not_called()
        with self.assertRaisesRegex(RuntimeError,'session boundary'):
            list(runtime.generate('a',None,raw_prompt=[1],max_tokens=1))

    def test_direct_and_scheduled_limits_before_ownership(self):
        direct=Runtime.__new__(Runtime);direct.context=16
        direct.session=Mock(side_effect=RuntimeError('session boundary'))
        scheduled=BatchedRuntime.__new__(BatchedRuntime);scheduled.context=16
        scheduled.models={'a':'unused'}
        scheduled.admission=Mock();scheduled.admission.acquire.side_effect=RuntimeError('admission boundary')
        for runtime in (direct,scheduled):
            for limit in (17,10**100):
                for options in ({'messages':[]},{'messages':None,'raw_prompt':[1]}):
                    with self.subTest(runtime=type(runtime).__name__,limit=limit,options=options),self.assertRaisesRegex(ValueError,'configured context'):
                        list(runtime.generate('a',max_tokens=limit,**options))
        direct.session.assert_not_called();scheduled.admission.acquire.assert_not_called()
        # Full-context output remains admissible for encoder-decoder requests;
        # prompt-specific checks remain at the existing later boundary.
        with self.assertRaisesRegex(RuntimeError,'session boundary'):
            list(direct.generate('a',None,max_tokens=16,raw_prompt=[1]))
        with self.assertRaisesRegex(RuntimeError,'admission boundary'):
            list(scheduled.generate('a',None,max_tokens=16,raw_prompt=[1]))

    def test_raw_prompt_limit_before_copy_or_ownership(self):
        from dllm.prompts import validate_raw
        direct=Runtime.__new__(Runtime);direct.context=16
        direct.session=Mock(side_effect=RuntimeError('session boundary'))
        scheduled=BatchedRuntime.__new__(BatchedRuntime);scheduled.context=16
        scheduled.models={'a':'unused'}
        scheduled.admission=Mock();scheduled.admission.acquire.side_effect=RuntimeError('admission boundary')
        class Oversized(list):
            def __iter__(self):raise AssertionError('oversized prompt iterated/copied')
        prompt=Oversized([1]*17)
        for runtime in (direct,scheduled):
            with self.subTest(runtime=type(runtime).__name__),self.assertRaisesRegex(ValueError,'raw prompt exceeds'):
                list(runtime.generate('a',None,max_tokens=1,raw_prompt=prompt))
            with self.subTest(scoring=type(runtime).__name__),self.assertRaisesRegex(ValueError,'raw prompt exceeds'):
                runtime.score('a',prompt,[1,2])
        direct.session.assert_not_called();scheduled.admission.acquire.assert_not_called()
        valid=[1]*16
        copied=validate_raw(valid,max_length=16)
        self.assertEqual(copied,valid);self.assertIsNot(copied,valid)
        for runtime,boundary in ((direct,'session boundary'),(scheduled,'admission boundary')):
            with self.assertRaisesRegex(RuntimeError,boundary):
                list(runtime.generate('a',None,max_tokens=1,raw_prompt=valid))
        self.assertEqual(validate_raw('long text',max_length=1),'long text')

    def test_embedding_rows_before_copy_and_ownership(self):
        from dllm.embeddings import options
        from dllm.encoder import EncoderWeights
        direct=Runtime.__new__(Runtime);direct.context=16
        direct.session=Mock(side_effect=RuntimeError('session boundary'))
        scheduled=BatchedRuntime.__new__(BatchedRuntime);scheduled.context=16
        scheduled.models={'a':'unused'}
        scheduled._submit=Mock(side_effect=RuntimeError('submission boundary'))
        encoder=EncoderWeights.__new__(EncoderWeights);encoder.max_context=16;encoder.task='embedding'
        class Oversized(list):
            def __iter__(self):raise AssertionError('oversized row iterated/copied')
        row=Oversized([1]*17)
        for inputs in (row,[row],[[1],row],[row,[1]]):
            for runtime in (direct,scheduled):
                with self.subTest(runtime=type(runtime).__name__),self.assertRaisesRegex(ValueError,'configured context'):
                    runtime.embed('a',inputs)
            with self.assertRaisesRegex(ValueError,'configured context'):encoder.encode(inputs)
        direct.session.assert_not_called();scheduled._submit.assert_not_called()
        valid=[1]*16
        for inputs in (valid,[valid],['hello']):
            for runtime,boundary in ((direct,'session boundary'),(scheduled,'submission boundary')):
                with self.assertRaisesRegex(RuntimeError,boundary):runtime.embed('a',inputs)
        copied=options([valid],max_length=16)[0]
        self.assertEqual(copied,[valid]);self.assertIsNot(copied[0],valid)

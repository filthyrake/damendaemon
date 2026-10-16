"""Impossible output requests cannot cause model activation or queue admission."""
import unittest
from unittest.mock import Mock
from dllm.runtime import Runtime
from dllm.scheduler import BatchedRuntime


class OutputPreflightTests(unittest.TestCase):
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

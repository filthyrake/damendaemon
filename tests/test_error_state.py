import gc
import unittest
import weakref
from dllm.error_state import detached_error


class ErrorStateTests(unittest.TestCase):
    def test_nested_failure_groups_detach_every_traceback(self):
        references=[]
        class Allocation:pass
        def failure():
            allocation=Allocation();references.append(weakref.ref(allocation))
            try:raise ValueError('worker fixture')
            except ValueError as error:
                error.add_note('child note')
                return ExceptionGroup('outer',[ExceptionGroup('inner',[error])])
        original=failure()
        detached=detached_error(original)
        original_child=original.exceptions[0].exceptions[0]
        child=detached.exceptions[0].exceptions[0]
        self.assertIsNot(child,original_child)
        self.assertIsNotNone(original_child.__traceback__)
        self.assertIsNone(child.__traceback__)
        self.assertEqual(child.__notes__,['child note'])
        self.assertIsNot(child.__notes__,original_child.__notes__)
        del original,original_child
        gc.collect()
        self.assertIsNone(references[0]())

    def test_fatal_group_preserves_base_exception_members(self):
        error=BaseExceptionGroup('fatal',[SystemExit('exit'),ValueError('request')])
        result=detached_error(error)
        self.assertIs(type(result),BaseExceptionGroup)
        self.assertIs(type(result.exceptions[0]),SystemExit)
        self.assertEqual(str(result.exceptions[0]),'exit')

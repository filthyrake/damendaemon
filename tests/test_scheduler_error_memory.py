import gc
import unittest
import weakref
from unittest.mock import patch
from damendaemon.scheduler import BatchedRuntime
from tests.test_scheduler import FakeRuntime


class SchedulerErrorMemoryTests(unittest.TestCase):
    def test_retained_nested_failure_does_not_pin_worker_allocations(self):
        references=[]
        class Allocation:pass
        class FailureRuntime(FakeRuntime):
            attempted=False
            def manage(self,**options):
                if self.attempted:return super().manage(**options)
                self.attempted=True
                allocation=Allocation();references.append(weakref.ref(allocation))
                try:raise ValueError('worker allocation fixture')
                except ValueError as error:
                    error.add_note('child diagnostic')
                    group=ExceptionGroup('operation failed',[ExceptionGroup('cleanup',[error])])
                    group.add_note('outer diagnostic')
                    raise group
        with patch('damendaemon.scheduler.Runtime',FailureRuntime):
            runtime=BatchedRuntime({'a':'fixture'})
            try:
                retained=None
                # #664: status is served as a direct snapshot, so the worker
                # command path is exercised with a mutating command.
                try:runtime.manage('unload','a')
                except ExceptionGroup as error:retained=error
                self.assertIsNotNone(retained)
                self.assertEqual(runtime.manage('status'),{'done':'status'})
                gc.collect()
                self.assertIsNone(references[0]())
                child=retained.exceptions[0].exceptions[0]
                self.assertIsInstance(child,ValueError)
                self.assertEqual(str(child),'worker allocation fixture')
                self.assertEqual(child.__notes__,['child diagnostic'])
                self.assertIsNone(child.__traceback__)
                self.assertIsNone(child.__cause__)
                self.assertIsNone(child.__context__)
                self.assertIn('outer diagnostic',retained.__notes__)
                self.assertTrue(any(note.startswith('Worker traceback:') for note in retained.__notes__))
                self.assertTrue(runtime.is_healthy())
            finally:runtime.close()

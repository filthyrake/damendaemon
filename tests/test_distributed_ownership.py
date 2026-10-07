from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import unittest
from unittest.mock import patch

from dllm.distributed_serving import DistributedRuntime,Request
from dllm.runtime import BusyError
from tests.test_distributed_serving import Pair,RuntimeHelpers


class OwnershipTests(RuntimeHelpers,unittest.TestCase):
    def test_pair_claim_is_atomic_and_invalid_construction_does_not_claim(self):
        pair=Pair()
        with self.assertRaises(ValueError):DistributedRuntime(pair,queue_size=0)
        self.assertIsNone(getattr(pair,'_serving_owner',None))
        barrier=threading.Barrier(2)
        def claim():
            barrier.wait()
            try:return DistributedRuntime(pair)
            except ValueError:return None
        with ThreadPoolExecutor(2) as pool:
            results=[f.result(2) for f in [pool.submit(claim) for _ in range(2)]]
        owners=[r for r in results if r is not None];self.assertEqual(len(owners),1)
        self.assertIs(pair._serving_owner,owners[0].owner);self.assertEqual(pair.calls,[])
        owners[0].close()
        with self.assertRaises(ValueError):DistributedRuntime(pair)

    def test_occupied_failed_closed_and_pending_pairs_are_rejected(self):
        for field,value in (('lengths',[2]),('failed',True),('closed',True),('pending',object())):
            pair=Pair();setattr(pair.state,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):DistributedRuntime(pair)
            self.assertIsNone(getattr(pair,'_serving_owner',None));self.assertEqual(pair.calls,[])

    def test_concurrent_loop_rejection_preserves_active_owner_and_rejects_restart(self):
        pair=Pair();runtime=DistributedRuntime(pair)
        with self.assertRaises(ValueError):runtime.run(heartbeat_interval=0)
        self.assertFalse(runtime.closed.is_set());self.assertIsNone(runtime.failure)
        _,errors=self.start(runtime)
        deadline=time.monotonic()+2
        while not runtime.execution_lock.locked() and time.monotonic()<deadline:time.sleep(.001)
        self.assertTrue(runtime.execution_lock.locked())
        with self.assertRaisesRegex(BusyError,'already has an owner'):runtime.run()
        result=list(runtime.generate('model',None,2,raw_prompt=[1]))
        self.assertEqual(result[-1]['usage']['completion_tokens'],2);self.assertFalse(errors)
        runtime.close()
        self.wait_closed(runtime)
        with self.assertRaisesRegex(BusyError,'closed'):runtime.run()
        self.assertIsNone(runtime.failure)

    def wait_closed(self,runtime):
        deadline=time.monotonic()+2
        while runtime.execution_lock.locked() and time.monotonic()<deadline:time.sleep(.001)
        self.assertFalse(runtime.execution_lock.locked())

    def test_primary_dispatch_error_survives_secondary_release_failure(self):
        pair=Pair();runtime=DistributedRuntime(pair)
        request=Request([1],2,time.monotonic(),queue.Queue(9));runtime.requests.put(request)
        # Unlike a poisoned transport, a local stage failure may leave its slot
        # release path eligible; a failing release must not replace that cause.
        with patch.object(pair,'step',side_effect=RuntimeError('primary dispatch')),patch.object(pair,'release',side_effect=RuntimeError('secondary release')):
            with self.assertRaisesRegex(RuntimeError,'primary dispatch') as error:runtime.run()
        self.assertIn('slot release failed',' '.join(error.exception.__notes__))
        self.assertIs(runtime.failure,error.exception);self.assertFalse(runtime.execution_lock.locked())
        self.assertIsInstance(request.events.get_nowait(),RuntimeError)

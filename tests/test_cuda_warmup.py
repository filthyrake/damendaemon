from contextlib import nullcontext
import threading
import unittest
from unittest.mock import Mock, patch

from dllm import cuda_warmup


class WarmupStreamTests(unittest.TestCase):
    def setUp(self):
        cuda_warmup._states.clear()
        self.addCleanup(cuda_warmup._states.clear)
        self.device = self.enterContext(patch('torch.cuda.current_device', return_value=0))
        self.create = self.enterContext(patch('torch.cuda.Stream', side_effect=lambda **kw: Mock()))
        self.current = self.enterContext(patch('torch.cuda.current_stream', return_value=Mock()))
        self.enterContext(patch('torch.cuda.stream', side_effect=lambda stream: nullcontext()))

    def test_reuses_stream_per_device_and_orders_caller(self):
        for _ in range(3):
            with cuda_warmup.warmup_stream():
                pass
        self.assertEqual(self.create.call_count, 1)
        stream = cuda_warmup._states[0][0]
        self.assertEqual(stream.wait_stream.call_count, 3)
        self.current.return_value.wait_stream.assert_called_with(stream)
        self.device.return_value = 1
        with cuda_warmup.warmup_stream():
            pass
        self.assertEqual(self.create.call_count, 2)
        self.assertIsNot(cuda_warmup._states[0][0], cuda_warmup._states[1][0])

    def test_exception_restores_ordering_and_releases_lock(self):
        with self.assertRaisesRegex(ValueError, 'warmup failed'):
            with cuda_warmup.warmup_stream():
                raise ValueError('warmup failed')
        stream = cuda_warmup._states[0][0]
        self.current.return_value.wait_stream.assert_called_once_with(stream)
        completed = threading.Event()

        def retry():
            with cuda_warmup.warmup_stream():
                completed.set()

        thread = threading.Thread(target=retry, daemon=True)
        thread.start()
        self.assertTrue(completed.wait(2))
        thread.join(2)

    def test_serializes_overlapping_warmups(self):
        attempted, entered = threading.Event(), threading.Event()

        def other():
            attempted.set()
            with cuda_warmup.warmup_stream():
                entered.set()

        with cuda_warmup.warmup_stream():
            thread = threading.Thread(target=other, daemon=True)
            thread.start()
            self.assertTrue(attempted.wait(2))
            self.assertFalse(entered.wait(.05))
        self.assertTrue(entered.wait(2))
        thread.join(2)


if __name__ == '__main__':
    unittest.main()

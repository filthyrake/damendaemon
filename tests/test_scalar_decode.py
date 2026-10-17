"""Single-request callers must use the executor's graph and authoritative slot length."""
import unittest
from unittest.mock import Mock,patch
from damendaemon.batch import BatchExecutor
from damendaemon.qwen2 import Executor


class ScalarDecodeTests(unittest.TestCase):
    def test_scalar_uses_one_row_graph_and_updates_length(self):
        executor=object.__new__(BatchExecutor)
        executor.batch_graphs={1:object()};executor.lengths=[7,19];executor.length=999
        def decode(tokens,slots):
            self.assertEqual((tokens,slots),([42],[0]));executor.lengths[0]+=1
            return 'logits','next'
        executor.decode_batch=Mock(side_effect=decode)
        self.assertEqual(executor.decode(42),('logits','next'))
        self.assertEqual(executor.length,8);self.assertEqual(executor.lengths[1],19)

    def test_no_graph_keeps_existing_eager_path(self):
        executor=object.__new__(BatchExecutor);executor.batch_graphs={}
        executor.decode_batch=Mock(side_effect=AssertionError('unexpected batch dispatch'))
        with patch.object(Executor,'decode',return_value=('eager','next')) as fallback:
            self.assertEqual(executor.decode(42),('eager','next'));fallback.assert_called_once_with(42)

import unittest
from unittest.mock import patch
import torch
from dllm.prefix import PrefixCache


class Executor:
    pass


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.executor=Executor()
        self.executor.kv=[torch.zeros(2,2,1,16,2)]
        self.executor.lengths=[0,0]
        self.cache=PrefixCache(self.executor,12,max_entries=2)
        memory=patch('dllm.prefix.available_memory',return_value=1024**3)
        memory.start();self.addCleanup(memory.stop)

    def remember(self,prompt,value=1):
        self.executor.kv[0][:,0].fill_(value)
        logits=torch.tensor([[float(value),0.]])
        self.cache.remember(prompt,0,logits)
        return logits

    @torch.inference_mode()
    def test_exact_restore_isolated_from_decode_and_returned_logits(self):
        original=self.remember([1,2,3],4)
        original.zero_()
        self.executor.kv[0].zero_()
        count,logits=self.cache.restore([1,2,3],1)
        self.assertEqual(count,3)
        self.assertEqual(logits.tolist(),[[4.,0.]])
        self.assertTrue((self.executor.kv[0][:,1,:,:3,:]==4).all())
        logits.zero_();self.executor.kv[0].zero_()
        self.assertEqual(self.cache.restore([1,2,3],0)[1].tolist(),[[4.,0.]])
        self.assertEqual(self.executor.lengths[0],3)

    def test_divergent_and_shorter_suffix_recompute_final_token(self):
        self.remember([1,2,3,4])
        self.assertEqual(self.cache.restore([1,2,9],1),(2,None))
        self.assertEqual(self.cache.restore([1,2],1),(1,None))
        self.assertEqual(self.cache.restore([9,2],1),(0,None))
        self.assertEqual(self.cache.restore([1,2,3,4,5],1),(4,None))

    def test_token_and_entry_bounds_follow_lru(self):
        self.remember([1]*6);self.remember([2]*6)
        self.cache.restore([1]*6,1)
        self.remember([3]*6)
        self.assertEqual(self.cache.tokens,12)
        self.assertEqual(self.cache.restore([2]*6,1),(0,None))
        self.assertEqual(len(self.cache.entries),2)
        self.remember([4]*16)
        self.assertEqual(self.cache.tokens,12)
        self.assertEqual(len(self.cache.entries),1)
        self.assertEqual(self.cache.restore([4]*16,1),(12,None))

    def test_low_device_memory_skips_snapshot(self):
        with patch('dllm.prefix.available_memory',return_value=0),patch('dllm.prefix.empty_cache'):
            self.remember([1,2,3])
        self.assertFalse(self.cache.entries)
        self.assertEqual(self.cache.tokens,0)

    def test_recurrent_snapshots_only_resume_complete_stored_prefix(self):
        self.executor.prefix_can_rewind=False
        self.remember([1,2,3,4])
        self.assertEqual(self.cache.restore([1,2,9],1),(0,None))
        self.assertEqual(self.cache.restore([1,2],1),(0,None))
        self.assertEqual(self.cache.restore([1,2,3,4,5],1),(4,None))
        self.assertEqual(self.cache.restore([1,2,3,4],1)[0],4)
        self.remember([9]*13)
        self.assertEqual(self.cache.tokens,4)

    def test_recurrent_snapshot_memory_uses_fixed_state_size(self):
        self.executor.prefix_can_rewind=False
        self.executor.prefix_state_bytes=lambda count:2*1024**3
        self.remember([1])
        self.assertFalse(self.cache.entries)

    def test_cache_does_not_own_executor_lifetime(self):
        import weakref
        reference=weakref.ref(self.executor)
        del self.executor
        self.assertIsNone(reference())

    def test_allocation_failure_discards_incomplete_snapshot(self):
        with patch.object(torch.Tensor,'clone',side_effect=torch.OutOfMemoryError('test')), patch('dllm.prefix.empty_cache'):
            self.remember([1,2,3])
        self.assertEqual(self.cache.tokens,0)
        self.assertFalse(self.cache.entries)

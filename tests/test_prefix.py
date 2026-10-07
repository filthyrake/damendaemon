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

    def test_scopes_isolate_exact_partial_logits_and_global_eviction(self):
        alice,bob = object(),object()
        prompt=[1,2,3,4]
        self.executor.kv[0][:,0].fill_(7)
        self.cache.remember(prompt,0,torch.tensor([[7.,0.]]),scope=alice)
        before=self.executor.kv[0][:,1].clone()
        self.assertEqual(self.cache.restore(prompt,1,scope=bob),(0,None))
        self.assertEqual(self.cache.restore([1,2,9],1,scope=bob),(0,None))
        self.assertEqual(self.cache.restore(prompt,1),(0,None))
        torch.testing.assert_close(self.executor.kv[0][:,1],before)
        self.assertEqual(self.executor.lengths[1],0)
        self.executor.kv[0][:,0].fill_(9)
        self.cache.remember(prompt,0,torch.tensor([[9.,0.]]),scope=bob)
        self.assertEqual(self.cache.tokens,8)
        self.assertEqual(self.cache.restore(prompt,1,scope=alice)[1].tolist(),[[7.,0.]])
        self.assertTrue((self.executor.kv[0][:,1,:,:4,:]==7).all())
        self.assertEqual(self.cache.restore(prompt,1,scope=bob)[1].tolist(),[[9.,0.]])
        self.assertEqual(self.cache.restore([1,2,9],1,scope=alice),(2,None))
        self.cache.remember([8]*6,0,torch.zeros(1,2),scope=alice)
        self.assertEqual(self.cache.tokens,10)
        self.assertEqual(self.cache.restore(prompt,1,scope=bob),(0,None))

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

    def test_scope_capacity_evicts_own_lru_before_global(self):
        from dllm.request_scope import CacheScope
        self.cache=PrefixCache(self.executor,32,max_entries=8)
        alice,bob=CacheScope(8),CacheScope(8)
        logits=torch.zeros(1,2)
        for scope,prompt in ((bob,[9]*4),(alice,[1]*4),(alice,[2]*4)):
            self.cache.remember(prompt,0,logits,scope=scope)
        self.cache.restore([1]*4,1,scope=alice)
        self.cache.remember([3]*4,0,logits,scope=alice)
        self.assertEqual(self.cache.restore([2]*4,1,scope=alice),(0,None))
        self.assertEqual(self.cache.restore([9]*4,1,scope=bob)[0],4)
        self.assertEqual(self.cache.restore([1]*4,1,scope=alice)[0],4)
        self.assertEqual(self.cache.tokens,12)
        self.cache.remember([7]*12,0,logits,scope=alice)
        self.assertEqual(self.cache.restore([7]*12,1,scope=alice),(8,None))
        self.assertEqual(self.cache.tokens,12)
        self.assertEqual(self.cache.restore([9]*4,1,scope=bob)[0],4)

    def test_scope_zero_recurrent_and_global_caps(self):
        from dllm.request_scope import CacheScope
        logits=torch.zeros(1,2)
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('disabled snapshot')):
            self.cache.remember([1,2],0,logits,scope=CacheScope(0))
        self.assertFalse(self.cache.entries)
        alice=CacheScope(3)
        self.executor.prefix_can_rewind=False
        self.cache.remember([1,2,3,4],0,logits,scope=alice)
        self.assertFalse(self.cache.entries)
        self.cache.remember([1,2,3],0,logits,scope=alice)
        self.assertEqual(self.cache.restore([1,2,3],1,scope=alice)[0],3)
        self.executor.prefix_can_rewind=True
        unlimited=CacheScope()
        self.cache.remember([7]*16,0,logits,scope=unlimited)
        self.assertEqual(self.cache.tokens,12)
        self.assertEqual(len(self.cache.entries),1)
        self.assertEqual(self.cache.restore([7]*16,1,scope=unlimited),(12,None))
        self.assertNotEqual(CacheScope(3),CacheScope(3))

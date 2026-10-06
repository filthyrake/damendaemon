"""Packed stage admission controls; real GPU numerical proof is separate."""
import unittest
from unittest.mock import Mock,patch
import torch
from dllm.layer_stage import LayerStage
from dllm.partition_policy import packed_prefill_enabled


class PackedStageTests(unittest.TestCase):
    def test_invalid_group_rejects_before_cache_or_gpu_work(self):
        stage=LayerStage.__new__(LayerStage)
        stage.context=8;stage.max_batch=4;stage.lengths=[0]*4
        stage.forward_packed_hidden=Mock();stage.embedding=Mock()
        for lengths,slots,rows in (([],[],0),([2],[0,1],2),([2,3],[0,0],5),
                                   ([2],[True],2),([True],[0],1),([9],[0],9),
                                   ([5,4],[0,1],9),([2],[4],2),([2],[0],3)):
            with self.subTest(lengths=lengths,slots=slots),self.assertRaises(ValueError):
                stage.run_prefill_packed(torch.empty(rows,dtype=torch.long),lengths,slots)
        self.assertEqual(stage.lengths,[0]*4)
        stage.forward_packed_hidden.assert_not_called();stage.embedding.assert_not_called()

    def test_explicit_policy_rejects_misspelled_values(self):
        for value,expected in (('0',False),('1',True)):
            with patch.dict('os.environ',{'DLLM_PARTITION_PACKED_PREFILL':value}):
                self.assertEqual(packed_prefill_enabled('q8_0'),expected)
        for value in ('true','yes','2',''):
            with patch.dict('os.environ',{'DLLM_PARTITION_PACKED_PREFILL':value}),self.assertRaises(ValueError):
                packed_prefill_enabled('q8_0')

    def test_default_packing_is_limited_to_clean_expanded_precision_gates(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertTrue(packed_prefill_enabled('none'))
            self.assertTrue(packed_prefill_enabled('q8_0'))
            self.assertFalse(packed_prefill_enabled('q4_0'))
        with patch.dict('os.environ',{'DLLM_PARTITION_PACKED_PREFILL':'1'}):
            self.assertTrue(packed_prefill_enabled('q4_0'))

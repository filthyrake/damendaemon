"""Storage and admission contracts; GPU numerical/performance proof is separate."""
import unittest
from unittest.mock import Mock,patch
import torch
from dllm.q4_stage import StageQ4Linear,repack_adjacent
from dllm.quantization import dequantize_q4
from dllm.partition_policy import q4_stage_adjacent_enabled


class Q4StageTests(unittest.TestCase):
    def test_permutation_preserves_every_nibble_and_signed_scale(self):
        canonical=torch.arange(256,dtype=torch.uint8).reshape(4,64)
        scales=torch.tensor([[0.,-2.,.5,1.]]*4,dtype=torch.float16)
        expected=dequantize_q4(canonical,scales)
        original=canonical.clone();packed=repack_adjacent(canonical)
        values=torch.stack((packed&15,packed>>4),dim=-1).reshape(4,4,32).float()-8
        actual=(values*scales.float().unsqueeze(-1)).reshape(4,128)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(canonical,original))
        self.assertEqual(packed.shape,canonical.shape)
        self.assertEqual(packed.dtype,canonical.dtype)

    def test_refresh_preserves_captured_storage_after_new_canonical_upload(self):
        target=torch.arange(32,dtype=torch.uint8).reshape(2,16)
        linear=StageQ4Linear.__new__(StageQ4Linear)
        linear.weights={id(target):target}
        pointer=target.data_ptr()
        for canonical in (target.clone(),torch.full_like(target,0x42)):
            target.copy_(canonical)
            with patch('dllm.q4_stage.repack_adjacent',wraps=repack_adjacent) as repack:
                linear.refresh()
                repack.assert_called_once()
            self.assertEqual(pointer,target.data_ptr())
            self.assertTrue(torch.equal(target,repack_adjacent(canonical)))

    def test_head_dispatch_is_unchanged(self):
        linear=StageQ4Linear.__new__(StageQ4Linear);linear.weights={};linear.base=Mock(return_value='head')
        x,head,bias=object(),object(),object()
        self.assertEqual(linear(x,head,bias),'head')
        linear.base.assert_called_once_with(x,head,bias)

    def test_policy_defaults_to_qualified_q4_and_allows_legacy_control(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertTrue(q4_stage_adjacent_enabled('q4_0'))
        with patch.dict('os.environ',{'DLLM_Q4_STAGE_ADJACENT':'1'}):
            self.assertTrue(q4_stage_adjacent_enabled('q4_0'))
            self.assertFalse(q4_stage_adjacent_enabled('q8_0'))
        with patch.dict('os.environ',{'DLLM_Q4_STAGE_ADJACENT':'0'}):
            self.assertFalse(q4_stage_adjacent_enabled('q4_0'))
        for value in ('true','','2'):
            with patch.dict('os.environ',{'DLLM_Q4_STAGE_ADJACENT':value}),self.assertRaises(ValueError):
                q4_stage_adjacent_enabled('q4_0')

    def test_runtime_bounds_keep_slow_down_tiles_static(self):
        from dllm.q4_stage import use_runtime_rows,runtime_rows_enabled
        for rows in (1,16,17,32,33,64,65,128):self.assertTrue(use_runtime_rows(rows,896))
        for rows in (1,16,33,64):self.assertTrue(use_runtime_rows(rows,4864))
        for rows in (0,17,32,65,128,129):self.assertFalse(use_runtime_rows(rows,4864))
        for rows,width in ((0,896),(129,896),(4,1024)):self.assertFalse(use_runtime_rows(rows,width))
        with patch.dict('os.environ',{},clear=True):self.assertFalse(runtime_rows_enabled())
        with patch.dict('os.environ',{'DLLM_Q4_STAGE_RUNTIME_ROWS':'1'}):self.assertTrue(runtime_rows_enabled())
        with patch.dict('os.environ',{'DLLM_Q4_STAGE_RUNTIME_ROWS':'invalid'}),self.assertRaises(ValueError):runtime_rows_enabled()

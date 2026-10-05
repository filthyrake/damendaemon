"""Admission and cache-lifecycle tests; no mocked GPU performance claims."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
from dllm.partition_policy import selected_split,validate
from dllm.partition import PartitionExecutor
from dllm.runtime import require_gpu


class PartitionTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(config=dict(model_type='qwen2',hidden_size=896,num_hidden_layers=24,
            num_attention_heads=14,num_key_value_heads=2,intermediate_size=4864,vocab_size=151936,head_dim=64),
            max_context=512,quantization='q8_0')

    def test_explicit_policy_and_distinct_visible_devices(self):
        with patch.dict('os.environ',{},clear=True):self.assertIsNone(selected_split())
        for value in ('0','-1','1.2','１２',''):
            with patch.dict('os.environ',{'DLLM_EXPERIMENTAL_LAYER_SPLIT':value},clear=True):
                with self.assertRaises(ValueError):selected_split()
        cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:2,set_device=Mock())
        with patch.dict('os.environ',{'DLLM_EXPERIMENTAL_LAYER_SPLIT':'12','CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b'},clear=True),patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda)}):
            require_gpu();cuda.set_device.assert_called_once_with(0)
        for visible in ('GPU-a','GPU-a,GPU-a','0,1'):
            with patch.dict('os.environ',{'DLLM_EXPERIMENTAL_LAYER_SPLIT':'12','CUDA_VISIBLE_DEVICES':visible},clear=True):
                with self.assertRaises(ValueError):require_gpu()
        with patch.dict('os.environ',{'DLLM_EXPERIMENTAL_LAYER_SPLIT':'12','DLLM_DEVICE':'mps'},clear=True):
            with self.assertRaises(ValueError):selected_split()

    def test_unqualified_modes_fail_before_gpu_construction(self):
        with patch.dict('os.environ',{},clear=True):
            model=self.model();validate(model,12,4)
            for split,batch in ((0,4),(11,4),(12,5)):
                with self.assertRaises(ValueError):validate(model,split,batch)
            model.quantization='q4_0'
            with self.assertRaises(ValueError):validate(model,12,4)
            model=self.model();model.max_context=513
            with self.assertRaises(ValueError):validate(model,12,4)
        for flag in ({'DLLM_KV_CACHE_TOKENS':'512'},{'DLLM_SPECULATIVE_TOKENS':'2'},{'DLLM_KERNELS':'torch'}):
            with patch.dict('os.environ',flag,clear=True):
                with self.assertRaises(ValueError):validate(self.model(),12,4)

    def executor(self):
        value=PartitionExecutor.__new__(PartitionExecutor)
        value.model=self.model();value.context=512;value.max_batch=4;value.lengths=[0]*4
        value.failed=False;value.closed=False;value.stages=[]
        return value

    def test_invalid_batches_cannot_mutate_cache_or_poison_other_requests(self):
        executor=self.executor();executor.lengths=[10,20,30,40]
        for prompts,slots in (([[1],[2]],[0,0]),([[1],[151936]],[0,1]),([[1]*513],[0]),([[1]],[True])):
            with self.assertRaises(ValueError):executor.prefill_batch(prompts,slots)
            self.assertFalse(executor.failed);self.assertEqual(executor.lengths,[10,20,30,40])
        executor.lengths[1]=0
        with self.assertRaises(ValueError):executor.decode_batch([1,2],[0,1])
        self.assertFalse(executor.failed)
        executor.failed=True
        with self.assertRaisesRegex(RuntimeError,'reactivation'):executor.decode_batch([1],[0])

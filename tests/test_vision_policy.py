"""Serving admission stays restricted while explicit backend validation expands."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from damendaemon.runtime import make_executor


class VisionPolicyTests(unittest.TestCase):
    def test_oversized_apple_batch_cannot_enter_serving_executor(self):
        model=SimpleNamespace(config={'model_type':'qwen3_5_text'},max_context=1024,
                              vision_weights=object(),scales={"quantized":object()})
        with patch.dict(os.environ,{'DAMENDAEMON_DEVICE':'mps'}), \
                patch('torch.backends.mps.is_available',return_value=True), \
                patch('damendaemon.vision_hybrid.VisionHybridExecutor') as constructor:
            with self.assertRaisesRegex(ValueError,'at most four slots'):
                make_executor(model,max_batch=5)
            constructor.assert_not_called()

    def test_rocm_admits_only_the_qualified_radeon(self):
        model=SimpleNamespace(config={'model_type':'qwen3_5_text'},max_context=1024,
                              vision_weights=object(),scales={"quantized":object()})
        # The qualified gfx1151 device (HOMER's Radeon 8060S) is admitted by name.
        with patch.dict(os.environ,{'DAMENDAEMON_DEVICE':'rocm'}), \
                patch('torch.cuda.is_available',return_value=True), \
                patch('torch.cuda.get_device_name',return_value='Radeon 8060S Graphics'), \
                patch('damendaemon.vision_hybrid.VisionHybridExecutor') as constructor:
            make_executor(model,max_batch=4)
            constructor.assert_called_once()
        # Any other ROCm device keeps the explicit error; nothing is executed.
        with patch.dict(os.environ,{'DAMENDAEMON_DEVICE':'rocm'}), \
                patch('torch.cuda.is_available',return_value=True), \
                patch('torch.cuda.get_device_name',return_value='AMD Radeon PRO W7900'), \
                patch('damendaemon.vision_hybrid.VisionHybridExecutor') as constructor:
            with self.assertRaisesRegex(ValueError,'Radeon 8060S'):
                make_executor(model,max_batch=4)
            constructor.assert_not_called()

    def test_cuda_admits_only_the_qualified_nvidia_devices(self):
        model=SimpleNamespace(config={'model_type':'qwen3_5_text'},max_context=1024,
                              vision_weights=object(),scales={"quantized":object()})
        # The qualified A6000 is admitted by name (#50, 2026-10-31).
        with patch.dict(os.environ,{'DAMENDAEMON_DEVICE':'cuda'}), \
                patch('torch.cuda.is_available',return_value=True), \
                patch('torch.cuda.get_device_name',return_value='NVIDIA RTX A6000'), \
                patch('damendaemon.vision_hybrid.VisionHybridExecutor') as constructor:
            make_executor(model,max_batch=4)
            constructor.assert_called_once()
        # Any other NVIDIA device keeps the explicit error; nothing is executed.
        with patch.dict(os.environ,{'DAMENDAEMON_DEVICE':'cuda'}), \
                patch('torch.cuda.is_available',return_value=True), \
                patch('torch.cuda.get_device_name',return_value='NVIDIA GeForce RTX 5090'), \
                patch('damendaemon.vision_hybrid.VisionHybridExecutor') as constructor:
            with self.assertRaisesRegex(ValueError,'A6000'):
                make_executor(model,max_batch=4)
            constructor.assert_not_called()


if __name__=='__main__':unittest.main()

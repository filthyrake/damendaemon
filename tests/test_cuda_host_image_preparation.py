from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import torch
from dllm.cuda_host_stage import CudaHostStageSlots
from dllm.glm_flash_host_image_generation import HostFlashImageGenerator


class CudaHostImagePreparationTests(unittest.TestCase):
    def test_failed_copy_retires_epoch_but_admission_rejection_does_not(self):
        # Failure ownership seam only; no accelerator arithmetic is mocked.
        for failed in (False,True):
            boundary=CudaHostStageSlots.__new__(CudaHostStageSlots)
            boundary.slots=SimpleNamespace(failed=failed,closed=failed)
            generator=HostFlashImageGenerator.__new__(HostFlashImageGenerator)
            generator.worker=SimpleNamespace(stage=boundary,_fail=Mock())
            error=RuntimeError('copy failed') if failed else ValueError('payload admission')
            with patch.object(boundary,'text_embeddings',side_effect=error):
                with self.assertRaises(type(error)):generator._text_embeddings(torch.tensor([1]))
            self.assertEqual(generator.worker._fail.call_count,int(failed))

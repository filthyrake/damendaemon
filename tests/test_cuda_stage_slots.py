import tempfile
from pathlib import Path
import unittest
from dllm.cuda_stage_slots import CudaGlmStageSlots,CudaFlashStageSlots,CudaDeepSeekStageSlots
from dllm import glm_reference_loader,glm_flash_reference_loader,deepseek_reference_loader
from tests import test_glm_reference_loader as glm_fixture,test_glm_flash_reference_loader as flash_fixture,test_deepseek_reference_loader as ds_fixture


class CudaStageSlotsTests(unittest.TestCase):
    def test_cpu_ranges_cannot_implicitly_select_cuda(self):
        for fixture,loader in ((glm_fixture,glm_reference_loader),(flash_fixture,glm_flash_reference_loader),(ds_fixture,deepseek_reference_loader)):
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp);fixture.fixture(path);source=loader.load_reference_stage(path,0,3)
                for kind in (CudaGlmStageSlots,CudaFlashStageSlots,CudaDeepSeekStageSlots):
                    with self.assertRaises(ValueError):kind(source,8,3)

import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_layer_stages import CudaGlmLayerStage,CudaFlashLayerStage
from tests.test_glm_model import model_fixture
from tests.test_glm_flash_model import fixture


class CudaStageAdmissionTests(unittest.TestCase):
    def test_cpu_blocks_rejected(self):
        for kind,model in ((CudaGlmLayerStage,model_fixture()),(CudaFlashLayerStage,fixture())):
            with self.assertRaises(ValueError):kind(model.blocks,start=0,total=len(model.blocks))

    def test_boundary_budget_precedes_payload_validation(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        for kind,core in ((CudaGlmLayerStage,'damendaemon.glm_layer_stage._GlmLayerStage'),(CudaFlashLayerStage,'damendaemon.glm_flash_layer_stage._FlashLayerStage')):
            stage=object.__new__(kind);stage.device=torch.device('cuda:0');stage.first=True;stage.last=False
            stage.dim=8;stage.max_output_bytes=1
            with FakeTensorMode(),patch(core+'._tokens',side_effect=AssertionError('payload before admission')):
                with self.assertRaisesRegex(ValueError,'budget'):stage._tokens(torch.empty(1,2,device='cuda',dtype=torch.int64),False)

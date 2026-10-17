import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_stage_placement import place_cuda_stage,_recipe
from tests import test_glm_reference_loader as glm_fixture,test_glm_flash_reference_loader as flash_fixture,test_deepseek_reference_loader as ds_fixture
from damendaemon import glm_reference_loader,glm_flash_reference_loader,deepseek_reference_loader


class CudaStagePlacementTests(unittest.TestCase):
    def test_loaded_ranges_preserve_cpu_recipe_and_reject_before_copy(self):
        for family,fixture,loader,ranges in (
                ('glm',glm_fixture,glm_reference_loader,((0,2),(2,3))),
                ('flash',flash_fixture,glm_flash_reference_loader,((0,1),(1,3))),
                ('deepseek',ds_fixture,deepseek_reference_loader,((0,1),(1,3)))):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as temp:
                path=Path(temp);fixture.fixture(path)
                for start,end in ranges:
                    source=loader.load_reference_stage(path,start,end)
                    module,options=_recipe(source,256*1024**2)
                    values=module._tensors(source,stage=True)
                    rebuilt=module._build(source,values,cuda=False,stage=True,**options)
                    self.assertEqual((rebuilt.start,rebuilt.end),(start,end))
                    self.assertEqual(rebuilt.embedding is None,start!=0)
                    self.assertEqual(rebuilt.head is None,end!=3)
                    self.assertIsNot(rebuilt.config,source.config)
                    with self.assertRaises(ValueError):module._tensors(source)
                    with patch('damendaemon.cuda_stage_placement._copy',side_effect=AssertionError('copy before admission')):
                        with self.assertRaises(ValueError):place_cuda_stage(source,device='cuda:0',max_bytes=1)
                        if source.last:
                            if family=='deepseek':source.head['norm.weight']=torch.ones(1)
                            else:source.norm=torch.ones(1)
                            with self.assertRaises(ValueError):place_cuda_stage(source,device='cuda:0',max_bytes=10**8)

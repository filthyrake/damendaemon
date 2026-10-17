import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from damendaemon.glm_flash_images import prepare_images,image_layout,validate_processor,MEAN,STD
from damendaemon.glm_flash_vision import FlashVision,vision_specs
from tests.test_glm_flash_vision import fixture


class FlashImageTests(unittest.TestCase):
    def setUp(self):
        old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)

    def test_pinned_processor_and_upstream_resize_cases(self):
        data=json.loads(Path('tests/fixtures/glm53_flash_image_contract.json').read_text())
        validate_processor(data['processor'])
        for case in data['resize_cases']:
            self.assertEqual(image_layout(*case['input']),tuple(case['canvas']+case['content']))
        for bad in ({},dict(data['processor'],patch_size=16),dict(data['processor'],do_rescale=1),dict(data['processor'],video=True)):
            with self.assertRaises(ValueError):validate_processor(bad)

    def test_scalar_patch_order_normalization_temporal_repeat_and_padding(self):
        image=(torch.arange(113*169*3).reshape(113,169,3)%256).to(torch.uint8)
        patches,grids=prepare_images([image]);self.assertEqual(grids,[[1,10,14]])
        row=0
        for block_h in range(0,10,2):
            for block_w in range(0,14,2):
                for ih in range(2):
                    for iw in range(2):
                        value=patches[row].reshape(3,2,14,14)
                        for channel,py,px in ((0,0,0),(1,7,9),(2,13,13)):
                            y=(block_h+ih)*14+py;x=(block_w+iw)*14+px
                            raw=int(image[y,x,channel]) if y<113 and x<169 else 0
                            expected=(raw/255.-MEAN[channel])/STD[channel]
                            self.assertAlmostEqual(float(value[channel,0,py,px]),expected,places=6)
                        torch.testing.assert_close(value[:,0],value[:,1],rtol=0,atol=0);row+=1

    def test_batch_order_upsample_and_encoder_integration(self):
        images=[torch.full((1,1,3),127,dtype=torch.uint8),torch.zeros(112,112,3,dtype=torch.uint8)]
        patches,grids=prepare_images(images)
        single=[prepare_images([image])[0] for image in images]
        torch.testing.assert_close(patches,torch.cat(single),rtol=0,atol=0)
        self.assertEqual(grids,[[1,8,8],[1,8,8]])
        c,_=fixture();c['patch_size']=14
        torch.manual_seed(109);weights={name:torch.randn(shape)*.02 for name,shape in vision_specs(c).items()}
        encoder=FlashVision(c,weights)
        actual=encoder.forward(patches,grids)
        expected=torch.cat([encoder.forward(value,[grid]) for value,grid in zip(single,grids)])
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
        normalized=single[0][0].reshape(3,2,14,14)
        for channel in range(3):self.assertAlmostEqual(float(normalized[channel,0,0,0]),(127/255.-MEAN[channel])/STD[channel],places=6)

    def test_resizing_uses_native_uint8_bicubic_before_normalization(self):
        from torch.nn import functional as F
        image=(torch.arange(17*29*3).reshape(17,29,3)%256).to(torch.uint8)
        h,w,ch,cw=image_layout(17,29);resize=F.interpolate
        reference=resize(image.permute(2,0,1).unsqueeze(0),size=(ch,cw),mode='bicubic',align_corners=False,antialias=True)[0]
        seen=[]
        def checked(value,**options):seen.append(value.dtype);return resize(value,**options)
        with patch('damendaemon.glm_flash_images.F.interpolate',side_effect=checked):patches,grids=prepare_images([image])
        self.assertEqual(seen,[torch.uint8]);rows,columns=grids[0][1:]
        for py,px in ((0,0),(min(ch-1,14),min(cw-1,21)),(h-1,w-1)):
            patch_y,patch_x=py//14,px//14
            index=((patch_y//2)*(columns//2)+patch_x//2)*4+(patch_y%2)*2+patch_x%2
            for channel in range(3):
                raw=int(reference[channel,py,px]) if py<ch and px<cw else 0
                value=patches[index].reshape(3,2,14,14)[channel,0,py%14,px%14]
                self.assertAlmostEqual(float(value),(raw/255.-MEAN[channel])/STD[channel],places=6)

    def test_complete_batch_admission_precedes_allocation_and_resize(self):
        image=torch.zeros(112,112,3,dtype=torch.uint8)
        with patch('damendaemon.glm_flash_images.torch.empty',side_effect=AssertionError('allocated')),patch('damendaemon.glm_flash_images.F.interpolate',side_effect=AssertionError('resized')):
            for options in (dict(max_patches=100),dict(max_output_bytes=1),dict(max_workspace_bytes=1)):
                with self.assertRaisesRegex(ValueError,'budget'):prepare_images([image,image],**options)
            for images in ([],[image,image.float()],[image,torch.zeros(0,3,3,dtype=torch.uint8)]):
                with self.assertRaises(ValueError):prepare_images(images)
        for shape in ((True,1),(0,1),(1,2**21)):
            with self.assertRaises(ValueError):image_layout(*shape)

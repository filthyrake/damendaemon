import unittest
from unittest.mock import patch

import torch
from dllm.vision_inputs import image_size, prepare_images


class VisionInputTests(unittest.TestCase):
    def test_budget_checked_before_any_resize(self):
        image=torch.zeros(192,320,3,dtype=torch.uint8)
        with patch('dllm.vision_inputs.F.interpolate') as resize:
            with self.assertRaisesRegex(ValueError,'budget'):
                prepare_images([image,image],max_patches=400)
            resize.assert_not_called()

    def test_rejects_invalid_inputs(self):
        for images in ([],[torch.zeros(256,256,3)],
                       [torch.zeros(0,256,3,dtype=torch.uint8)],
                       [torch.zeros(1,201,3,dtype=torch.uint8)]):
            with self.subTest(images=[tuple(x.shape) for x in images]):
                with self.assertRaises(ValueError):prepare_images(images)
        for dimensions in ((0,32),(32,-1),(True,32)):
            with self.assertRaises(ValueError):image_size(*dimensions)

    def test_input_unchanged_and_images_independent(self):
        black=torch.zeros(256,256,3,dtype=torch.uint8)
        white=torch.full_like(black,255)
        pixels,grids=prepare_images([black,white])
        self.assertEqual(grids,[[1,16,16],[1,16,16]])
        self.assertTrue(bool((pixels[:256]==-1).all()))
        self.assertTrue(bool((pixels[256:]==1).all()))
        self.assertTrue(bool((black==0).all()))
        self.assertTrue(bool((white==255).all()))


if __name__=='__main__':unittest.main()

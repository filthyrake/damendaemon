import unittest

import torch
from dllm.vision_positions import image_positions, rotary_tables


class VisionPositionTests(unittest.TestCase):
    def test_rejects_missing_and_extra_features(self):
        for tokens,grids in (([1],[[1,2,2]]),([248056],[]),
                             ([248056,1],[[1,4,4]]),([248056],[[2,2,2]]),
                             ([248056],[[1,3,2]])):
            with self.subTest(tokens=tokens,grids=grids):
                with self.assertRaises(ValueError):image_positions(tokens,grids)

    def test_decode_offset_does_not_change_physical_length(self):
        tokens=[1]+[248056]*64+[2]
        positions,offset=image_positions(tokens,[[1,16,16]])
        self.assertEqual(len(tokens),66)
        self.assertEqual(offset,-56)
        self.assertEqual(positions[:,-1].tolist(),[9,9,9])
        self.assertEqual(len(tokens)+offset,10)

    def test_text_only_remains_ordinary_rotary(self):
        positions,offset=image_positions([11,12,13],[])
        self.assertEqual(offset,0)
        self.assertTrue(torch.equal(positions,torch.arange(3).expand(3,-1)))
        cos,sin=rotary_tables(positions)
        self.assertTrue(bool((cos[0]==1).all()))
        self.assertTrue(bool((sin[0]==0).all()))


if __name__=='__main__':unittest.main()

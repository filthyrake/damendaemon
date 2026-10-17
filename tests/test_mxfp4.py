import unittest
import torch
from damendaemon.mxfp4 import validate,unpack


class Mxfp4Tests(unittest.TestCase):
    def test_layout_preserves_expert_and_output_axes(self):
        blocks=torch.zeros((3,17,9,16),dtype=torch.uint8)
        scales=torch.zeros((3,17,9),dtype=torch.uint8)
        self.assertEqual(validate(blocks,scales),(3,17,288))
        self.assertEqual(validate(blocks[0].contiguous(),scales[0].contiguous()),(17,288))

    def test_malformed_layouts_fail_before_native_dispatch(self):
        blocks=torch.zeros((2,7,3,16),dtype=torch.uint8)
        scales=torch.zeros((2,7,3),dtype=torch.uint8)
        for b,s in ((blocks.float(),scales),(blocks,scales.float()),(blocks,scales[:1]),
                (blocks[...,:15],scales),(blocks.transpose(0,1),scales.transpose(0,1)),
                (blocks[:0],scales[:0])):
            with self.assertRaises(ValueError):validate(b,s)
        with self.assertRaises(ValueError):unpack(blocks,scales)


if __name__=='__main__':unittest.main()

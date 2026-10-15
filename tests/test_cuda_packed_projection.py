import unittest
from unittest.mock import patch
import torch
from dllm.cuda_packed_projection import cuda_packed_linear
from dllm.packed_tiles import validate_layout


class CudaPackedAdmissionTests(unittest.TestCase):
    def test_cpu_inputs_rejected_before_decode(self):
        with patch('dllm.cuda_packed_projection._decode_tile',side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError,'CUDA'):
                cuda_packed_linear(torch.ones(1,32),torch.zeros(3,16,dtype=torch.uint8),
                                   torch.full((3,1),127,dtype=torch.uint8),format='e2m1')

    def test_fake_cuda_metadata_empty_and_cross_device(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            weight=torch.empty(3,16,device='cuda:0',dtype=torch.uint8)
            scales=torch.empty(3,1,device='cuda:0',dtype=torch.uint8)
            hidden=torch.empty(0,32,device='cuda:0')
            self.assertEqual(validate_layout(weight,scales,format='e2m1',device_type='cuda'),(3,32))
            with patch('dllm.cuda_packed_projection._decode_tile',side_effect=AssertionError('decoded')):
                self.assertEqual(cuda_packed_linear(hidden,weight,scales,format='e2m1').shape,(0,3))
                with self.assertRaises(ValueError):cuda_packed_linear(hidden,weight,scales,format='e2m1',workspace_bytes=1)
                with self.assertRaises(ValueError):cuda_packed_linear(hidden.to('cuda:1'),weight,scales,format='e2m1')
                with self.assertRaises(ValueError):cuda_packed_linear(hidden,weight,scales.to('cuda:1'),format='e2m1')
                with self.assertRaises(ValueError):cuda_packed_linear(hidden,weight,scales,format='e2m1',row_range=(1,4))
            # CPU API stays CPU-only even though byte arithmetic is shared.
            with self.assertRaises(ValueError):validate_layout(weight,scales,format='e2m1')


if __name__=='__main__':unittest.main()

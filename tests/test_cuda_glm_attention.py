import unittest
from unittest.mock import patch
import torch
from damendaemon.cuda_prepared_projection import CudaPackedProjection
from damendaemon.cuda_glm_attention import project_mla,sparse_latent_attention
from damendaemon.glm_attention import project_mla as cpu_project, _projection


class CudaGlmAttentionTests(unittest.TestCase):
    def test_cpu_entrypoint_stays_cpu_and_cuda_rejects_cpu(self):
        with self.assertRaises(ValueError):project_mla(torch.ones(1,1,6),torch.zeros(1,1,dtype=torch.int64),{},heads=2,nope_dim=2,rope_dim=4)
        with self.assertRaises(ValueError):sparse_latent_attention(torch.ones(1),None,None,None,None,None,nope_dim=2,value_dim=3)
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            with self.assertRaises(ValueError):cpu_project(torch.ones(1,1,6,device='cuda:0'),torch.zeros(1,1,dtype=torch.int64,device='cuda:0'),{},heads=2,nope_dim=2,rope_dim=4)

    def test_cuda_adapter_row_coordinates_empty_and_device_contract(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            weight=torch.empty(133,160,dtype=torch.uint8,device='cuda:0')
            scales=torch.empty(2,2,dtype=torch.float32,device='cuda:0')
            original=CudaPackedProjection(weight,scales,format='e4m3fn_f32')
            selected=original.rows(125,133).rows(1,6)
            self.assertEqual((selected.start,selected.stop),(126,131))
            self.assertIs(selected.weight,weight);self.assertIs(selected.scales,scales)
            self.assertFalse(_projection(selected));self.assertTrue(_projection(selected,'cuda:0'))
            with patch('damendaemon.cuda_prepared_projection.cuda_packed_linear',side_effect=AssertionError('executed')):
                self.assertEqual(selected.linear(torch.empty(0,3,160,device='cuda:0')).shape,(0,3,5))
                for hidden in (torch.empty(1,160),torch.empty(1,160,device='cuda:1'),torch.empty(1,160,dtype=torch.float16,device='cuda:0')):
                    with self.assertRaises(ValueError):selected.linear(hidden)
            with self.assertRaises(ValueError):CudaPackedProjection(weight,scales.cpu(),format='e4m3fn_f32')


if __name__=='__main__':unittest.main()

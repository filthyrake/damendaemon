import unittest
from unittest.mock import patch
import torch
from dllm.cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
from dllm.expert_ffn import activate_expert


class CudaPackedExpertTests(unittest.TestCase):
    def matrices(self,device='cuda:0'):
        return [CudaPackedMatrix(torch.empty(32,16,device=device,dtype=torch.uint8),
                  torch.empty(32,1,device=device,dtype=torch.uint8),'e2m1') for _ in range(3)]

    def test_fake_cuda_admission_and_empty(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            matrices=self.matrices();expert=CudaPackedExpert(*matrices)
            with patch.object(expert,'_row',side_effect=AssertionError('executed')):
                self.assertEqual(expert.forward(torch.empty(0,32,device='cuda:0')).shape,(0,32))
                for hidden in (torch.empty(1,32),torch.empty(1,31,device='cuda:0'),torch.empty(1,32,device='cuda:1')):
                    with self.assertRaises(ValueError):expert.forward(hidden)
                with self.assertRaises(ValueError):expert.forward(torch.empty(1,32,device='cuda:0'),route_weights=torch.ones(1,1))
            for budget in (1,True,-1):
                with self.assertRaises(ValueError):CudaPackedExpert(*matrices,workspace_bytes=budget)
            with self.assertRaises(ValueError):CudaPackedExpert(matrices[0],matrices[1],self.matrices('cuda:1')[2])
            with self.assertRaises(ValueError):CudaPackedExpert(*matrices,activation_limit=float('nan'))

    def test_cpu_storage_rejected(self):
        with self.assertRaises(ValueError):CudaPackedExpert(*self.matrices('cpu'))

    def test_row_cast_clipping_and_weighting_order(self):
        # CPU arithmetic checks the control order only, not CUDA execution.
        expert=CudaPackedExpert.__new__(CudaPackedExpert)
        expert.gate,expert.up,expert.down='gate','up','down';expert.activation_limit=.7
        hidden=torch.tensor([[.25,-.75]],dtype=torch.float16)
        coefficient=torch.tensor([[.37]],dtype=torch.float32)
        gate=torch.tensor([[1.001,-2.003]]);up=torch.tensor([[2.007,-3.009]])
        calls=[]
        def project(value,matrix):
            calls.append((matrix,value.clone()))
            return {'gate':gate,'up':up}.get(matrix,value.float()*2)
        with patch.object(expert,'_project',side_effect=project):result=expert._row(hidden,coefficient)
        expected=activate_expert(gate.half().float(),up.half().float(),coefficient,.7).half()
        torch.testing.assert_close(calls[-1][1],expected,rtol=0,atol=0)
        torch.testing.assert_close(result,(expected.float()*2).half(),rtol=0,atol=0)


if __name__=='__main__':unittest.main()

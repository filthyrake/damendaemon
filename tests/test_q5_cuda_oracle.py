import os
import unittest
from unittest.mock import patch
import torch
from scripts.experiments.q5_cuda_oracle import require_guard, verify_cuda_output
from scripts.experiments.q5_precision_replay import oracle_kinds


class CudaOracleTests(unittest.TestCase):
    def test_refuses_unowned_or_ambiguous_device(self):
        for env in ({}, {'CUDA_VISIBLE_DEVICES':'0','DAMENDAEMON_SHARED_CUDA_GUARD':'1'},
                    {'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b','DAMENDAEMON_SHARED_CUDA_GUARD':'1'},
                    {'CUDA_VISIBLE_DEVICES':'GPU-a'}):
            with patch.dict(os.environ,env,clear=True), self.assertRaises(ValueError):
                require_guard()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-a','DAMENDAEMON_SHARED_CUDA_GUARD':'1'},clear=True):
            require_guard()

    def test_refuses_cpu_or_non_tensor_arithmetic(self):
        for output in (torch.ones(1),(torch.ones(1),None),None):
            with self.assertRaises(ValueError):
                verify_cuda_output(output)

    def test_oracle_kind_requires_completed_explicit_gpu_proof(self):
        cpu=dict(reference_device='cpu',oracles={'float16':{},'float32':{}})
        self.assertEqual(oracle_kinds(cpu),('float16','float32'))
        gpu=dict(reference_device='cuda',completed=True,offload='accelerate.cpu_offload',
                 oracles={'cuda_float16':{}})
        self.assertEqual(oracle_kinds(gpu),('cuda_float16',))
        for change in (dict(completed=False),dict(offload='cpu'),dict(oracles={'../x':{}}),dict(reference_device='cpu')):
            with self.assertRaises(ValueError):
                oracle_kinds(dict(gpu,**change))


if __name__ == '__main__':
    unittest.main()

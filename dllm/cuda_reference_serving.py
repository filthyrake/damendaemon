"""Explicit serialized prepared CUDA GLM HTTP runtime; no default CLI admission."""
from .reference_serving import ReferenceRuntime
from .cuda_glm_generation import CudaGlmToolGenerator


class CudaGlmReferenceRuntime(ReferenceRuntime):
    runner_types=(CudaGlmToolGenerator,)
    scope='cuda_prepared_reference'
    backend_suffix='cuda-prepared-reference'

    def __init__(self,runner,alias='model'):
        if type(runner) is not CudaGlmToolGenerator or runner.device.type!='cuda':
            raise ValueError('CUDA HTTP requires an explicit prepared CUDA GLM tool runner')
        self.device=str(runner.device)
        super().__init__(runner,alias)

    def execution_status(self):
        return dict(execution='prepared_cuda_glm',performance_qualified=False)

    def manage(self,action,**options):
        result=super().manage(action,**options)
        result.update(device=self.device,scope=self.scope)
        result['models']=[dict(id=self.alias,residency='closed' if self.closed else 'cuda')]
        return result

    def model_manifest(self):
        raise ValueError('prepared CUDA reference cannot be exported as a qualified production manifest')

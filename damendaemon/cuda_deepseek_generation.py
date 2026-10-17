"""Explicit single-sequence prepared CUDA DeepSeek native text/tool generation."""
from .deepseek_generation import ReferenceGenerator
from .cuda_deepseek_model import CudaDeepSeekModel


class CudaDeepSeekGenerator(ReferenceGenerator):
    model_type=CudaDeepSeekModel
    device_type='cuda'

"""Explicit CUDA slot owners; cursor/slot metadata stays on the CPU."""
from .glm_stage_slots import GlmStageSlots
from .glm_flash_stage_slots import FlashStageSlots
from .deepseek_stage_slots import DeepSeekStageSlots
from .cuda_layer_stages import CudaGlmLayerStage,CudaFlashLayerStage
from .cuda_deepseek_layer_stage import CudaDeepSeekLayerStage


class _CudaSlots:
    """Reuse ownership and family equations, without implicit boundary copies.

    Callers provide CUDA activations/routing IDs and CPU positions/slot IDs.
    Aggregate cache admission occurs before any slot allocation. These are not
    CPU host-worker stages; transport requires a separately admitted adapter.
    """
    @property
    def actual_device(self):return self.stage.device


class CudaGlmStageSlots(_CudaSlots,GlmStageSlots):
    stage_type=CudaGlmLayerStage


class CudaFlashStageSlots(_CudaSlots,FlashStageSlots):
    stage_type=CudaFlashLayerStage


class CudaDeepSeekStageSlots(_CudaSlots,DeepSeekStageSlots):
    stage_type=CudaDeepSeekLayerStage

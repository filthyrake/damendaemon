"""Opt-in local CUDA pipeline construction; no serving or GPU qualification."""
import os

import torch

from .hybrid_layout import choose_stage_splits
from .hybrid_pipeline import HybridPipeline
from .hybrid_stage import HybridLayerStage, shard_model


class PipelineBuildError(RuntimeError):
    """Own failed stages until close succeeds; never silently drop live buffers."""
    def __init__(self, message, stages, devices):
        super().__init__(message)
        self.stages = tuple(stages)
        self.devices = tuple(devices)
        self.closed = False

    def close(self):
        if self.closed:
            return
        # All devices must finish before any completed/partial stage releases.
        for device in self.devices:
            torch.cuda.synchronize(device)
        for stage in self.stages:
            stage.release_partial()
        for device in self.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        self.closed = True


def build_cuda_pipeline(model, devices, budgets, max_batch=4):
    """Plan before allocation, using explicit byte caps plus live free memory.

    Returns the internal coordinator. Synchronous unquantized FP16 CUDA text is
    admitted by default; packed Q8/Q4 hybrid weights require the explicit
    DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT opt-in (Q4 additionally
    requires compensated precision and builds Triton stages per the qualified
    A30 numeric path). A build failure raises PipelineBuildError; if its closed
    flag is false, the caller must retain it and retry close once the device
    failure is resolved.
    """
    from .hybrid_pipeline_policy import quant_pipeline_enabled
    from .layer_stream import enabled as offload_enabled
    quant_pipeline = quant_pipeline_enabled()
    if (os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE') != '1'
            or os.environ.get('DAMENDAEMON_DEVICE', 'cuda') != 'cuda'
            or os.environ.get('DAMENDAEMON_HYBRID_PRECISION', 'float16') not in (
                ('float16',) if not quant_pipeline else ('float16', 'compensated'))
            or (quant_pipeline and model.quantization == 'q4_0'
                and os.environ.get('DAMENDAEMON_HYBRID_PRECISION') != 'compensated')
            or os.environ.get('DAMENDAEMON_HYBRID_PREFILL', 'scan') != 'scan'
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT') is not None
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_VISION', '0') != '0'
            or os.environ.get('DAMENDAEMON_KV_CACHE_TOKENS', '0') != '0'
            or offload_enabled() or getattr(torch.version, 'hip', None)):
        raise ValueError('hybrid CUDA pipeline requires explicit opt-in, FP16/compensated text without offload, legacy splitting or paged KV')
    if (type(max_batch) is not int or not 1 <= max_batch <= 4
            or type(model.max_context) is not int
            or not 1 <= model.max_context <= min(32768, model.config['max_position_embeddings'])):
        raise ValueError('hybrid CUDA pipeline requires context <=32768 and batch 1..4')
    if (not isinstance(devices, (list, tuple)) or not devices
            or any(type(d) is not int or d < 0 for d in devices)
            or len(set(devices)) != len(devices)
            or not isinstance(budgets, (list, tuple)) or len(budgets) != len(devices)):
        raise ValueError('provide distinct CUDA ordinals and one byte budget per device')
    # Validate ownership and CPU weights without changing or allocating them.
    shard_model(model, 0, model.config['num_hidden_layers'])
    placement = choose_stage_splits(model.weights, model.config, model.max_context,
                                    max_batch, 2, budgets, scales=model.scales)
    if not placement['storage_plan_found']:
        raise MemoryError(placement['reason'])
    if not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count():
        raise ValueError('pipeline CUDA ordinal is unavailable')
    # Plan again against observed free memory; explicit caps are never raised.
    limits = [min(limit, torch.cuda.mem_get_info(device)[0]) for device, limit in zip(devices, budgets)]
    if any(limit <= 0 for limit in limits):
        raise MemoryError('no free memory on a pipeline device')
    placement = choose_stage_splits(model.weights, model.config, model.max_context,
                                    max_batch, 2, limits, scales=model.scales)
    if not placement['storage_plan_found']:
        raise MemoryError(placement['reason'])
    stages, touched = [], []
    failure = None
    try:
        for device, budget in zip(devices, placement['stages']):
            with torch.cuda.device(device):
                if torch.cuda.mem_get_info(device)[0] < budget['required_memory_bytes']:
                    raise MemoryError('pipeline device headroom changed during construction')
                # Retain ownership even if __init__ fails halfway through a copy.
                stage = HybridLayerStage.__new__(HybridLayerStage)
                stages.append(stage)
                touched.append(device)
                stage.__init__(model, budget['layer_start'], budget['layer_end'], max_batch,
                               kernels='triton' if model.quantization != 'none' else 'torch')
        pipeline = HybridPipeline(stages)
        pipeline.model = model
        pipeline.placement = placement
        pipeline.devices = tuple(devices)
        for stage in stages:
            stage.synchronize()
        return pipeline
    except BaseException as error:
        # Do not retain the original traceback's temporary allocation references.
        failure = PipelineBuildError(f'pipeline construction failed: {type(error).__name__}: {error}',
                                     stages, touched)
    try:
        failure.close()
    except BaseException as error:
        failure.add_note(f'cleanup incomplete; retain this error and retry close: {type(error).__name__}: {error}')
    raise failure

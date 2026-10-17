"""Globally budgeted construction of an explicit packed Flash CPU pipeline."""
from .reference_pipeline_build import build_reference_pipeline,snapshot_inventory as _inventory
from .glm_flash_reference_loader import _load_reference,load_reference_stage
from .glm_flash_layer_stage import validate_range
from .glm_flash_layer_pipeline import FlashLayerPipeline


def _inspect(path,**options):return _load_reference(path,None,None,inspect_only=True,**options)


def _ranges(config,splits):
    total=config['text_config']['num_hidden_layers'];boundaries=(0,*splits,total)
    ranges=tuple(zip(boundaries,boundaries[1:]))
    for start,end in ranges:validate_range(start,end,total)
    return ranges


def load_reference_pipeline(path,layer_splits,*,max_weight_bytes=256*1024**2,
                            projection_workspace_bytes=8*1024**2,
                            max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    options=dict(max_weight_bytes=max_weight_bytes,projection_workspace_bytes=projection_workspace_bytes,
                 max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_output_bytes)
    return build_reference_pipeline(path,layer_splits,inspect_snapshot=_inspect,ranges_for=_ranges,
        load_stage=load_reference_stage,pipeline_type=FlashLayerPipeline,options=options,inventory=_inventory)

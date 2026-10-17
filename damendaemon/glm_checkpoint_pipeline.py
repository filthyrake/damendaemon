"""Globally budgeted construction of a local packed GLM stage pipeline."""
from .reference_pipeline_build import build_reference_pipeline,snapshot_inventory as _inventory
from .glm_reference_loader import _load_reference,load_reference_stage
from .glm_layer_stage import validate_range
from .glm_layer_pipeline import GlmLayerPipeline


def _inspect(path,**options):return _load_reference(path,None,inspect_only=True,**options)


def _ranges(config,splits):
    boundaries=(0,*splits,config['num_hidden_layers']);ranges=tuple(zip(boundaries,boundaries[1:]))
    for start,end in ranges:validate_range(start,end,config['num_hidden_layers'],config['indexer_types'])
    return ranges


def load_reference_pipeline(path,layer_splits,*,max_weight_bytes=256*1024**2,
                            projection_workspace_bytes=8*1024**2,
                            max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Audit total retained/conversion weights before range-local payload loading."""
    options=dict(max_weight_bytes=max_weight_bytes,projection_workspace_bytes=projection_workspace_bytes,
                 max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_output_bytes)
    return build_reference_pipeline(path,layer_splits,inspect_snapshot=_inspect,ranges_for=_ranges,
        load_stage=load_reference_stage,pipeline_type=GlmLayerPipeline,options=options,inventory=_inventory)

"""Header-only declared storage audit for explicit CUDA host range plans."""
from .reference_stage_plan import ReferenceStorageLayout,_cache_per_layer
from .host_topology import HostTopology


def audit_cuda_host_storage(layout,ranges,host_budgets,device_budgets,*,context=512,slots=4,chunk_size=128,
        max_weight_bytes=256*1024**2,max_device_bytes=256*1024**2,max_cache_bytes=256*1024**2,
        device_workspace_bytes=256*1024**2,max_output_bytes=64*1024**2,max_transfer_bytes=64*1024**2,
        frame_bytes=64*1024**2,input_kind='tokens'):
    """Check unchanged cuts/caps; never infer free memory or admit execution.

    Budgets are caller-declared after reserving unrelated workloads. Device bytes
    match the unique prepared tensor recipe, including shared DeepSeek frequency
    vectors per range. Source conversion and all CPU preparation frequencies are
    separately counted on the host. Explicit workspace/output/transfer ceilings
    are allowances, not measured or reserved peaks. No device probe is performed.
    """
    if not isinstance(layout,ReferenceStorageLayout):raise ValueError('CUDA storage audit requires inspected headers')
    controls=(max_weight_bytes,max_device_bytes,max_cache_bytes,device_workspace_bytes,max_output_bytes,max_transfer_bytes)
    if (any(type(v) is not int or not 1<=v<=2**31 for v in controls)
            or type(frame_bytes) is not int or not 256<=frame_bytes<=64*1024**2):
        raise ValueError('invalid CUDA storage audit caps')
    layout.verify_unchanged();root=layout.config
    topology=HostTopology(root,ranges,bytes(32),quantization='fp4_fp8_reference' if layout.family=='deepseek' else 'fp8_reference',
        context=context,context_limit=2**20,slots=slots,chunk_size=chunk_size,frame_bytes=64*1024**2,input_kind=input_kind)
    for budgets in (host_budgets,device_budgets):
        if (not isinstance(budgets,(list,tuple)) or len(budgets)!=len(topology.ranges)
                or any(type(v) is not int or not 1<=v<=2**40 for v in budgets)):
            raise ValueError('CUDA storage audit needs one positive host/device budget per range')
    c=topology.text_config;count=c['num_hidden_layers'];width=topology.width
    rows=max(slots,min(context,chunk_size));activation=4*rows*width
    token_frame=8*rows if layout.family=='deepseek' else 0
    caches=_cache_per_layer(layout.family,c,context)
    # The CPU loader prepares all ratio vectors, but each stage keeps/copies only
    # vectors referenced by its owned blocks. Equal ratios share one tensor.
    frequency_bytes=(c['qk_rope_head_dim']//2*4) if layout.family=='deepseek' else 0
    all_frequencies=frequency_bytes*len(set(c['compress_ratios'][:count])) if frequency_bytes else 0
    projection,component,_=layout.inspection_limits
    stages=[]
    for index,((start,end),host_limit,device_limit) in enumerate(zip(topology.ranges,host_budgets,device_budgets)):
        components=list(layout.components[start+1:end+1])
        if start==0:components.append(layout.components[0])
        if end==count:components.append(layout.components[-1])
        retained=sum(v[0] for v in components);conversion=max(v[1] for v in components)
        derived=frequency_bytes*len(set(c['compress_ratios'][start:end])) if frequency_bytes else 0
        device_payload=retained+derived;cache=slots*sum(caches[start:end])
        output=max(activation,4*slots*c['vocab_size'] if end==count else 0)
        transfer=max(2*activation+token_frame,8*rows+activation if start==0 else 0,
            4*slots*(width+c['vocab_size']) if end==count else 0)
        if start==0 and input_kind=='flash_embeddings':
            # Image preparation embeds a whole admitted prompt before chunked
            # host dispatch. The Flash slot validates its expanded stream extent
            # even though the adapter transfers only collapsed embedding rows.
            output=max(output,4*context*width)
            transfer=max(transfer,context*(8+4*c['hidden_size']))
        host_allowance=projection+component+max_output_bytes+max_transfer_bytes
        device_allowance=device_workspace_bytes+max_output_bytes+max_transfer_bytes
        host_required=retained+conversion+all_frequencies+host_allowance
        device_required=device_payload+cache+device_allowance
        failures=[]
        for name,used,cap in (('loader_weight',retained+conversion,max_weight_bytes),('device_payload',device_payload,max_device_bytes),
                ('cache',cache,max_cache_bytes),('output',output,max_output_bytes),('transfer',transfer,max_transfer_bytes),
                ('frame',max(activation,token_frame),frame_bytes),('host_budget',host_required,host_limit),
                ('device_budget',device_required,device_limit)):
            if used>cap:failures.append(name)
        stages.append(dict(index=index,layers=[start,end],selected_parameter_bytes=retained,source_conversion_overlap_bytes=conversion,
            derived_device_frequency_bytes=derived,host_preparation_frequency_bytes=all_frequencies,device_payload_bytes=device_payload,
            cache_reservation_bytes=cache,required_output_bytes=output,required_transfer_payload_bytes=transfer,
            host_allowance_bytes=host_allowance,device_allowance_bytes=device_allowance,
            required_declared_host_bytes=host_required,required_declared_device_bytes=device_required,
            host_budget_bytes=host_limit,device_budget_bytes=device_limit,exceeded_caps=failures))
    report=dict(scope='cuda_host_declared_storage',family=layout.family,plan={'ranges':[list(p) for p in topology.ranges]},
        context=context,slots=slots,chunk_size=chunk_size,input_kind=input_kind,metadata_id=layout.metadata_id,
        source_model_identity_verified=False,execution_admitted=False,device_compatibility_verified=False,
        peak_memory_measured=False,declared_limits_fit=all(not stage['exceeded_caps'] for stage in stages),stages=stages,
        excluded_memory=['Python/header objects','allocator rounding and fragmentation','CUDA driver and context',
            'unmeasured arithmetic and preparation temporaries','other workloads','coordinator vision and full image prompt storage'],
        limits=dict(loader_weight_bytes=max_weight_bytes,device_payload_bytes=max_device_bytes,cache_bytes=max_cache_bytes,
            device_workspace_allowance_bytes=device_workspace_bytes,output_allowance_bytes=max_output_bytes,
            transfer_payload_allowance_bytes=max_transfer_bytes,frame_bytes=frame_bytes))
    layout.verify_unchanged();return report

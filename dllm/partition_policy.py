"""Explicit, narrow admission for the qualified two-device layer partition."""
import os


def rebind_enabled():
    value=os.environ.get('DLLM_PARTITION_REBIND','1')
    if value not in ('0','1'):raise ValueError('DLLM_PARTITION_REBIND must be 0 or 1')
    return value=='1'


def q4_stage_adjacent_enabled(quantization):
    value=os.environ.get('DLLM_Q4_STAGE_ADJACENT','1')
    if value not in ('0','1'):raise ValueError('DLLM_Q4_STAGE_ADJACENT must be 0 or 1')
    return quantization=='q4_0' and value=='1'


def packed_prefill_enabled(quantization):
    value=os.environ.get('DLLM_PARTITION_PACKED_PREFILL')
    if value is None:return quantization in ('none','q8_0') or q4_stage_adjacent_enabled(quantization)
    if value not in ('0','1'):raise ValueError('DLLM_PARTITION_PACKED_PREFILL must be 0 or 1')
    return value=='1'


def selected_split():
    value=os.environ.get('DLLM_EXPERIMENTAL_LAYER_SPLIT')
    if value is None:return None
    if not value.isascii() or not value.isdecimal() or int(value)<=0:
        raise ValueError('DLLM_EXPERIMENTAL_LAYER_SPLIT must be a positive layer index')
    if os.environ.get('DLLM_DEVICE','cuda')!='cuda':
        raise ValueError('layer partition requires CUDA')
    return int(value)


def validate(model,split,max_batch):
    c=model.config
    # Keep serving behind the geometry, precision and limits qualified on real
    # GPUs. The experimental harness may investigate other partition points.
    geometry={'model_type':'qwen2','hidden_size':896,'num_hidden_layers':24,
              'num_attention_heads':14,'num_key_value_heads':2,'intermediate_size':4864,
              'vocab_size':151936,'head_dim':64}
    if any(c.get(key)!=value for key,value in geometry.items()):
        raise ValueError('layer partition serving is qualified only for Qwen2.5-0.5B geometry')
    if split!=12 or not 1<=max_batch<=4 or not 1<=model.max_context<=512:
        raise ValueError('layer partition serving requires split 12, context <=512, batch <=4')
    if (model.quantization not in ('none','q8_0','q4_0')
            or getattr(model,'weight_format',model.quantization)!=model.quantization
            or getattr(model,'packed_formats',{})
            or getattr(model,'awq_matrices',{}) or c.get('rope_scaling')):
        raise ValueError('layer partition requires standard RoPE and FP16, Q8 or Q4 weights')
    if (os.environ.get('DLLM_KV_CACHE_TOKENS') is not None
            or os.environ.get('DLLM_SPECULATIVE_TOKENS','0')!='0'
            or os.environ.get('DLLM_KERNELS','triton') not in ('auto','triton')):
        raise ValueError('layer partition requires native kernels and fixed KV slots without speculation')

"""Metadata-only Qwen3.5-MoE main-text checkpoint contract (#14 Qwen target).

Transcribed from the Qwen3.6-35B-A3B multimodal MoE checkpoint (root
model_type qwen3_5_moe / text model_type qwen3_5_moe_text; 26 BF16
safetensors shards, 1,045 index tensors): 693 main-text tensors with 19 MTP
and 333 visual tensors explicitly excluded, not validated or loaded. Routed
experts ship stacked as two fused 3-D tensors per layer
(``mlp.experts.gate_up_proj`` [experts, 2*inter, hidden] and
``mlp.experts.down_proj`` [experts, hidden, inter]) beside a router
(``mlp.gate.weight``), a dense shared expert and a [1, hidden]
``shared_expert_gate`` combine gate. The hybrid linear/full attention tensor
inventory matches the qualified dense Qwen3.5 text contract; the MoE FFN is
the delta. No weight payloads are read; routing semantics, model execution
and quality are not qualified here. Checkpoint pinning hashes live in
tests/fixtures/qwen3_5_moe_contract.json.
"""
from .checkpoint_contract import TensorSpec


ROOT = 'model.language_model.'

DTYPES = {'floating': frozenset(('BF16', 'F32'))}


def _integer(config, key, *, low=1, high=2**21):
    value = config.get(key)
    if type(value) is not int or not low <= value <= high:
        raise ValueError('invalid qwen3.5-MoE integer field: ' + key)
    return value


def _exact(config, key, expected):
    if config.get(key) != expected or type(config.get(key)) is not type(expected):
        raise ValueError(f'unsupported qwen3.5-MoE {key} contract value')


def tensor_specs(config):
    """Validate the multimodal root + text geometry and transcribe the tensor inventory."""
    if not isinstance(config, dict) or config.get('model_type') != 'qwen3_5_moe':
        raise ValueError('expected a qwen3_5_moe multimodal root configuration')
    text = config.get('text_config')
    if not isinstance(text, dict) or text.get('model_type') != 'qwen3_5_moe_text':
        raise ValueError('expected a qwen3_5_moe_text text configuration')
    if not isinstance(config.get('vision_config'), dict):
        raise ValueError('qwen3_5_moe root requires its vision configuration container')
    if config.get('tie_word_embeddings') is not False:
        raise ValueError('qwen3.5-MoE contract requires an untied output head')

    dim = _integer(text, 'hidden_size')
    vocab = _integer(text, 'vocab_size')
    layers = _integer(text, 'num_hidden_layers')
    heads = _integer(text, 'num_attention_heads')
    kv = _integer(text, 'num_key_value_heads', low=1)
    head = _integer(text, 'head_dim')
    lk = _integer(text, 'linear_num_key_heads')
    lv = _integer(text, 'linear_num_value_heads')
    dk = _integer(text, 'linear_key_head_dim')
    dv = _integer(text, 'linear_value_head_dim')
    kernel = _integer(text, 'linear_conv_kernel_dim')
    interval = _integer(text, 'full_attention_interval')
    experts = _integer(text, 'num_experts')
    top = _integer(text, 'num_experts_per_tok')
    inter = _integer(text, 'moe_intermediate_size')
    shared = _integer(text, 'shared_expert_intermediate_size')
    mtp = _integer(text, 'mtp_num_hidden_layers', low=0, high=16)
    if kv > heads or top > experts:
        raise ValueError('qwen3.5-MoE head or expert routing bounds are inconsistent')

    schedule = text.get('layer_types')
    if (not isinstance(schedule, list) or len(schedule) != layers
            or any(kind not in ('linear_attention', 'full_attention') for kind in schedule)
            or any((kind == 'full_attention') != ((index + 1) % interval == 0)
                   for index, kind in enumerate(schedule))):
        raise ValueError('qwen3.5-MoE layer schedule does not match its full-attention interval')

    _exact(text, 'attention_bias', False)
    _exact(text, 'attn_output_gate', True)
    _exact(text, 'hidden_act', 'silu')
    _exact(text, 'mamba_ssm_dtype', 'float32')
    if text.get('dtype') not in ('bfloat16', 'float32'):
        raise ValueError('qwen3.5-MoE contract supports bfloat16/float32 checkpoints only')
    eps = text.get('rms_norm_eps')
    if type(eps) not in (int, float):
        raise ValueError('invalid qwen3.5-MoE normalization epsilon')
    if not 0 < eps < float('inf'):
        raise ValueError('invalid qwen3.5-MoE normalization epsilon')
    partial = text.get('partial_rotary_factor')
    if type(partial) not in (int, float):
        raise ValueError('invalid qwen3.5-MoE partial rotary factor')
    if not 0 < partial <= 1:
        raise ValueError('invalid qwen3.5-MoE partial rotary factor')
    rope = text.get('rope_parameters')
    section = rope.get('mrope_section') if isinstance(rope, dict) else None
    if (not isinstance(rope, dict) or rope.get('mrope_interleaved') is not True
            or not isinstance(section, list) or len(section) != 3
            or any(type(value) is not int or value < 1 for value in section)
            or head * partial != 2 * sum(section)):
        raise ValueError('qwen3.5-MoE interleaved mrope sections do not cover the rotary width')

    specs = {}

    def add(name, shape):
        if name in specs:
            raise ValueError('duplicate qwen3.5-MoE tensor spec: ' + name)
        specs[name] = TensorSpec(tuple(shape), 'floating')

    add(ROOT + 'embed_tokens.weight', (vocab, dim))
    add(ROOT + 'norm.weight', (dim,))
    add('lm_head.weight', (vocab, dim))
    width = 2 * lk * dk + lv * dv
    for layer, kind in enumerate(schedule):
        root = ROOT + f'layers.{layer}.'
        add(root + 'input_layernorm.weight', (dim,))
        add(root + 'post_attention_layernorm.weight', (dim,))
        if kind == 'full_attention':
            # attn_output_gate packs the per-head gate into q_proj (2*heads*head_dim rows).
            for name, rows in (('q_proj', 2 * heads * head), ('k_proj', kv * head), ('v_proj', kv * head)):
                add(root + f'self_attn.{name}.weight', (rows, dim))
            add(root + 'self_attn.o_proj.weight', (dim, heads * head))
            add(root + 'self_attn.q_norm.weight', (head,))
            add(root + 'self_attn.k_norm.weight', (head,))
        else:
            for name, rows in (('in_proj_qkv', width), ('in_proj_z', lv * dv),
                               ('in_proj_b', lv), ('in_proj_a', lv)):
                add(root + f'linear_attn.{name}.weight', (rows, dim))
            add(root + 'linear_attn.out_proj.weight', (dim, lv * dv))
            add(root + 'linear_attn.conv1d.weight', (width, 1, kernel))
            add(root + 'linear_attn.dt_bias', (lv,))
            add(root + 'linear_attn.A_log', (lv,))
            add(root + 'linear_attn.norm.weight', (dv,))
        add(root + 'mlp.gate.weight', (experts, dim))
        add(root + 'mlp.experts.gate_up_proj', (experts, 2 * inter, dim))
        add(root + 'mlp.experts.down_proj', (experts, dim, inter))
        add(root + 'mlp.shared_expert.gate_proj.weight', (shared, dim))
        add(root + 'mlp.shared_expert.up_proj.weight', (shared, dim))
        add(root + 'mlp.shared_expert.down_proj.weight', (dim, shared))
        add(root + 'mlp.shared_expert_gate.weight', (1, dim))
    del mtp  # read for validation only; MTP tensors are excluded from the inventory
    return specs


def audit_names(config, names):
    specs = tensor_specs(config)
    names = list(names)
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError('qwen3.5-MoE tensor names must be distinct strings')
    mtp = sorted(name for name in names if name.startswith('mtp.'))
    vision = sorted(name for name in names if name.startswith('model.visual.'))
    main = set(names) - set(mtp) - set(vision)
    missing = set(specs) - main
    extra = main - set(specs)
    if missing or extra:
        raise ValueError(f'qwen3.5-MoE text inventory mismatch: missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]}')
    return dict(main_tensors=len(specs), excluded_mtp=mtp, excluded_vision=vision)


def audit_headers(config, headers):
    if not isinstance(headers, dict):
        raise ValueError('qwen3.5-MoE headers must be a mapping')
    report = audit_names(config, headers)
    for name, spec in tensor_specs(config).items():
        entry = headers[name]
        if (not isinstance(entry, dict) or not isinstance(entry.get('shape'), (list, tuple))
                or any(type(value) is not int for value in entry['shape']) or tuple(entry['shape']) != spec.shape
                or not isinstance(entry.get('dtype'), str) or entry['dtype'] not in DTYPES[spec.storage]):
            raise ValueError(f'qwen3.5-MoE tensor shape/storage mismatch: {name}')
    return report


def audit_snapshot(path, *, names_only=False):
    from .checkpoint_contract import audit_snapshot as audit
    return audit(path, tensor_specs, audit_names, audit_headers, names_only=names_only)

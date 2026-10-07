"""Offline main-model tensor contract for the pinned DeepSeek V4 Flash layout.

Names/geometry follow official inference/model.py and the tensor index at
60d8d70770c6776ff598c94bb586a859a38244f1 (MIT). No model admission or weight
materialization. MTP tensors are explicitly excluded and reported, not validated.
"""
from .checkpoint_contract import TensorSpec


DTYPES = {
    'floating': frozenset(('BF16', 'F32')),
    'fp8': frozenset(('F8_E4M3', 'F8_E4M3FN')),
    'fp4': frozenset(('I8', 'U8')),
    'scale': frozenset(('F8_E8M0', 'F8_E8M0FNU')),
    'integer': frozenset(('I32', 'I64')),
}


def tensor_specs(config):
    """Describe physical checkpoint shapes, including packed widths and scales."""
    if not isinstance(config, dict) or config.get('model_type') != 'deepseek_v4':
        raise ValueError('expected a DeepSeek V4 configuration')
    fields = ('hidden_size', 'vocab_size', 'num_hidden_layers', 'n_routed_experts',
              'num_experts_per_tok', 'moe_intermediate_size', 'hc_mult', 'num_attention_heads',
              'head_dim', 'q_lora_rank', 'o_groups', 'o_lora_rank', 'index_n_heads', 'index_head_dim')
    if any(type(config.get(key)) is not int or not 1 <= config[key] <= 2**21 for key in fields):
        raise ValueError('invalid DeepSeek positive integer geometry')
    dim, vocab, layers, experts, top, intermediate, hc, heads, head, qrank, groups, orank, ih, idim = (config[k] for k in fields)
    hashes = config.get('num_hash_layers')
    ratios = config.get('compress_ratios')
    mtp_layers = config.get('num_nextn_predict_layers', 0)
    if (type(hashes) is not int or not 0 <= hashes <= layers or not isinstance(ratios, list)
            or type(mtp_layers) is not int or not 0 <= mtp_layers <= 16
            or len(ratios) not in (layers, layers+mtp_layers) or any(type(r) is not int or r not in (0, 4, 128) for r in ratios)
            or top > experts or heads % groups or dim % 32 or intermediate % 32
            or layers*(6*experts+100)+6 > 1_000_000):
        raise ValueError('unsupported hash/compression/expert geometry or tensor inventory size')
    if (config.get('expert_dtype') != 'fp4' or type(config.get('n_shared_experts')) is not int
            or config['n_shared_experts'] != 1):
        raise ValueError('checkpoint contract requires FP4 routed experts and one shared expert')
    quant = config.get('quantization_config')
    if (not isinstance(quant, dict) or quant.get('quant_method') != 'fp8' or quant.get('fmt') != 'e4m3'
            or quant.get('scale_fmt') != 'ue8m0' or quant.get('weight_block_size') != [128, 128]):
        raise ValueError('checkpoint contract requires E4M3 weights and E8M0 128x128 block scales')
    specs = {}
    def add(name, shape, storage='floating'):
        specs[name] = TensorSpec(tuple(shape), storage)
    def linear(name, output, inputs, packed=False):
        if packed:
            add(name+'.weight', (output, inputs//2), 'fp4')
            add(name+'.scale', (output, inputs//32), 'scale')
        else:
            add(name+'.weight', (output, inputs), 'fp8')
            add(name+'.scale', ((output+127)//128, (inputs+127)//128), 'scale')
    def compressor(name, ratio, width):
        channels = width*(2 if ratio == 4 else 1)
        add(name+'.ape', (ratio, channels))
        add(name+'.wkv.weight', (channels, dim))
        add(name+'.wgate.weight', (channels, dim))
        add(name+'.norm.weight', (width,))
    add('embed.weight', (vocab, dim))
    add('head.weight', (vocab, dim))
    add('norm.weight', (dim,))
    add('hc_head_fn', (hc, hc*dim))
    add('hc_head_base', (hc,))
    add('hc_head_scale', (1,))
    for layer, ratio in enumerate(ratios[:layers]):
        prefix = f'layers.{layer}'
        for part in ('attn', 'ffn'):
            add(f'{prefix}.hc_{part}_fn', (hc*(hc+2), hc*dim))
            add(f'{prefix}.hc_{part}_base', (hc*(hc+2),))
            add(f'{prefix}.hc_{part}_scale', (3,))
            add(f'{prefix}.{part}_norm.weight', (dim,))
        attn = prefix+'.attn'
        add(attn+'.attn_sink', (heads,))
        linear(attn+'.wq_a', qrank, dim)
        linear(attn+'.wq_b', heads*head, qrank)
        add(attn+'.q_norm.weight', (qrank,))
        linear(attn+'.wkv', head, dim)
        add(attn+'.kv_norm.weight', (head,))
        linear(attn+'.wo_a', groups*orank, heads*head//groups)
        linear(attn+'.wo_b', dim, groups*orank)
        if ratio:
            compressor(attn+'.compressor', ratio, head)
        if ratio == 4:
            indexer = attn+'.indexer'
            linear(indexer+'.wq_b', ih*idim, qrank)
            add(indexer+'.weights_proj.weight', (ih, dim))
            compressor(indexer+'.compressor', ratio, idim)
        ffn = prefix+'.ffn'
        add(ffn+'.gate.weight', (experts, dim))
        if layer < hashes:
            add(ffn+'.gate.tid2eid', (vocab, top), 'integer')
        else:
            add(ffn+'.gate.bias', (experts,))
        for expert in range(experts+1):
            name = ffn+('.shared_experts' if expert == experts else f'.experts.{expert}')
            linear(name+'.w1', intermediate, dim, expert != experts)
            linear(name+'.w3', intermediate, dim, expert != experts)
            linear(name+'.w2', dim, intermediate, expert != experts)
    return specs


def audit_names(config, names):
    """Check main-model inventory; return the explicit, unqualified MTP exclusions."""
    specs = tensor_specs(config)
    names = list(names)
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError('checkpoint tensor names must be distinct strings')
    excluded = sorted(name for name in names if name.startswith('mtp.'))
    main = set(names)-set(excluded)
    missing, extra = sorted(set(specs)-main), sorted(main-set(specs))
    if missing or extra:
        raise ValueError(f'checkpoint main-model inventory mismatch: missing={missing[:4]}, extra={extra[:4]}')
    return {'main_tensors':len(specs), 'excluded_mtp':excluded}


def audit_headers(config, headers):
    """Validate caller-supplied header metadata without reading tensor values."""
    if not isinstance(headers, dict):
        raise ValueError('checkpoint headers must be a tensor metadata mapping')
    result = audit_names(config, headers)
    for name, spec in tensor_specs(config).items():
        entry = headers[name]
        if (not isinstance(entry, dict) or not isinstance(entry.get('shape'), (list, tuple))
                or any(type(size) is not int for size in entry['shape'])
                or tuple(entry['shape']) != spec.shape or not isinstance(entry.get('dtype'), str)
                or entry['dtype'] not in DTYPES[spec.storage]):
            raise ValueError(f'checkpoint tensor shape/storage mismatch: {name}')
    return result


def audit_snapshot(path, *, names_only=False):
    """Audit local metadata; names-only mode needs no shard files at all."""
    from .checkpoint_contract import audit_snapshot as audit
    return audit(path,tensor_specs,audit_names,audit_headers,names_only=names_only)

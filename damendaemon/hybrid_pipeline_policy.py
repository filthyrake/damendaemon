"""Explicit serving admission for the experimental local CUDA pipeline."""
import os


def budgets():
    flag = os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE', '0')
    text = os.environ.get('DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB')
    if flag not in ('0', '1'):
        raise ValueError('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE must be 0 or 1')
    if flag == '0':
        if text is not None:
            raise ValueError('pipeline budgets require explicit pipeline opt-in')
        return None
    entries = text.split(',') if text is not None else []
    if (not 1 <= len(entries) <= 16 or any(not value.isascii() or not value.isdecimal()
                                         or int(value) <= 0 for value in entries)):
        raise ValueError('set DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB to 1..16 comma-separated positive integers')
    return [int(value)*1024**2 for value in entries]


def quant_pipeline_enabled():
    value = os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT', '0')
    if value not in ('0', '1'):
        raise ValueError('DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT must be 0 or 1')
    return value == '1'


def validate_environment(use_graph=False):
    limits = budgets()
    if limits is None:
        return
    quant_pipeline = quant_pipeline_enabled()
    precision = os.environ.get('DAMENDAEMON_HYBRID_PRECISION', 'float16')
    if (os.environ.get('DAMENDAEMON_DEVICE', 'cuda') != 'cuda'
            or use_graph or precision not in (('float16',) if not quant_pipeline
                                              else ('float16', 'compensated'))
            or os.environ.get('DAMENDAEMON_HYBRID_PREFILL', 'scan') != 'scan'
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD', '0') != '0'
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT') is not None
            or os.environ.get('DAMENDAEMON_EXPERIMENTAL_VISION', '0') != '0'
            or os.environ.get('DAMENDAEMON_KV_CACHE_TOKENS') is not None):
        raise ValueError('pipeline serving requires unquantized CUDA hybrid FP16 text and --no-graph, without offload, vision or paged KV')


def validate(config, quantization, context, batch=1, use_graph=False):
    limits = budgets()
    if limits is None:
        return
    validate_environment(use_graph)
    quant_pipeline = quant_pipeline_enabled()
    if config.get('model_type') != 'qwen3_5_text' or quantization not in (
            ('none',) if not quant_pipeline else ('none', 'q8_0', 'q4_0')):
        raise ValueError('pipeline serving requires unquantized hybrid text weights unless the '
                         'explicit quantized-pipeline opt-in is set')
    if quant_pipeline and quantization == 'q4_0' and \
            os.environ.get('DAMENDAEMON_HYBRID_PRECISION', 'float16') != 'compensated':
        raise ValueError('Q4 pipeline stages require compensated precision (the qualified A30 Q4 numeric path)')
    if (type(context) is not int or not 1 <= context <= min(2048, config['max_position_embeddings'])
            or type(batch) is not int or not 1 <= batch <= 4
            or len(limits) > config['num_hidden_layers']):
        raise ValueError('pipeline serving requires context <=2048, batch 1..4 and at least one layer per device')

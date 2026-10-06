"""Validation-only reference loader; never imported by the serving runtime."""
import json
from pathlib import Path


def roundtrip_reference_weight(weight, quantization, rows_per_chunk=128):
    """Use upstream GGUF conversion with bounded FP32/NumPy temporaries."""
    import torch
    from gguf.quants import quantize, dequantize
    from gguf.constants import GGMLQuantizationType
    if rows_per_chunk < 1:
        raise ValueError('rows_per_chunk must be positive')
    kind = getattr(GGMLQuantizationType, quantization.upper())
    source = weight.detach()
    restored = torch.empty(source.shape, dtype=source.dtype, device='cpu')
    for start in range(0, source.shape[0], rows_per_chunk):
        block = source[start:start + rows_per_chunk].float().cpu().numpy()
        decoded = dequantize(quantize(block, kind), kind)
        restored[start:start + rows_per_chunk].copy_(torch.from_numpy(decoded))
    return restored


def load_reference(path, reference_quantization='none', **kwargs):
    config = json.loads((Path(path) / 'config.json').read_text())
    if config['model_type'] == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(path, **kwargs)
    if reference_quantization in ('q8_0','q4_0'):
        import torch
        # Independent upstream quantizer; replace each Linear parameter so a
        # tied output head does not also quantize the input embedding.
        for name, module in model.named_modules():
            # This text reference follows the runtime's text-only packing.
            # Vision and MTP weights are neither loaded nor quantized by that path.
            if config['model_type'] == 'qwen3_5' and (
                    name.startswith('model.visual.') or name.startswith('mtp.')):
                continue
            if isinstance(module, torch.nn.Linear):
                # Mamba-130M's rank-48 time-step projection does not form whole
                # Q8/Q4 blocks. dllm retains that small matrix in FP32.
                if config['model_type']=='mamba' and module.weight.shape[1]%32:
                    continue
                restored = roundtrip_reference_weight(module.weight, reference_quantization)
                module.weight = torch.nn.Parameter(restored, requires_grad=False)
    elif reference_quantization != 'none':
        raise ValueError('unsupported reference quantization')
    return model

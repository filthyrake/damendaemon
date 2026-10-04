"""Validation-only reference loader; never imported by the serving runtime."""
import json
from pathlib import Path


def load_reference(path, reference_quantization='none', **kwargs):
    config = json.loads((Path(path) / 'config.json').read_text())
    if config['model_type'] == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM
        cls = AutoModelForCausalLM
    model = cls.from_pretrained(path, **kwargs)
    if reference_quantization == 'q8_0':
        import torch
        from gguf.quants import quantize, dequantize
        from gguf.constants import GGMLQuantizationType
        # Independent upstream quantizer; replace each Linear parameter so a
        # tied output head does not also quantize the input embedding.
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach().float().cpu().numpy()
                packed = quantize(weight, GGMLQuantizationType.Q8_0)
                restored = dequantize(packed, GGMLQuantizationType.Q8_0)
                module.weight = torch.nn.Parameter(torch.from_numpy(restored).to(module.weight.dtype), requires_grad=False)
    elif reference_quantization != 'none':
        raise ValueError('unsupported reference quantization')
    return model

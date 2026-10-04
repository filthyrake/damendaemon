"""Validation-only reference loader; never imported by the serving runtime."""
import json
from pathlib import Path


def load_reference(path, **kwargs):
    config = json.loads((Path(path) / 'config.json').read_text())
    if config['model_type'] == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM
        cls = AutoModelForCausalLM
    return cls.from_pretrained(path, **kwargs)

"""Canonical model locations without losing split-checkpoint sibling names."""
from pathlib import Path
import re


def canonical_model_path(value):
    path = Path(value).expanduser()
    # Hugging Face snapshot links point at extensionless, unrelated blob names.
    # Split GGUF discovery must retain the first shard's numbered namespace.
    if re.fullmatch(r'.+-00001-of-[0-9]{5}\.gguf', path.name, re.IGNORECASE):
        return path.parent.resolve()/path.name
    return path.resolve()

"""Compare imported Llama rotary frequencies with an independent HF formula."""
import argparse
import json

import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from damendaemon.gguf_model import plan, read_tensor
from damendaemon.rotary import inverse_frequencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('hf_config')
    parser.add_argument('--position', type=int, default=2047)
    args = parser.parse_args()
    source = plan(args.model)
    actual = inverse_frequencies(source.config, 'cpu')
    with open(args.model, 'rb') as handle:
        factors, _ = read_tensor(handle, source.tensors['rope_freqs.weight'])
    actual = actual / factors
    config = AutoConfig.from_pretrained(args.hf_config, local_files_only=True, trust_remote_code=False)
    reference = LlamaRotaryEmbedding(config=config, device='cpu').inv_freq
    error = actual.double() - reference.double()
    print(json.dumps(dict(model=args.model, hf_config=args.hf_config, position=args.position,
        imported_frequencies=actual.tolist(), reference_frequencies=reference.tolist(), factors=factors.tolist(),
        max_absolute_frequency_error=error.abs().max().item(),
        max_relative_frequency_error=(error.abs()/reference.double().abs().clamp_min(1e-30)).max().item(),
        max_phase_error=(error*args.position).abs().max().item()), indent=2))


if __name__ == '__main__':
    main()

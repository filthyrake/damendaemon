"""Isolated Qwen2.5-0.5B Metal Q4 validation; public admission stays unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import torch
from damendaemon.config import decoder_config
from damendaemon.qwen2 import ModelWeights
from damendaemon.quantization import Linear
from damendaemon import quantized_metal_q4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.environ.get('DAMENDAEMON_DEVICE') != 'mps' or os.environ.get('DAMENDAEMON_QUANTIZATION') != 'q4_0':
        parser.error('explicit MPS and Q4 environment required')
    report = dict(completed=False, selected_calls=0, fallback_calls=0,
                  scope='isolated Metal Q4 full-model experiment; public runtime admission unchanged',
                  source_sha256=hashlib.sha256(Path(quantized_metal_q4.__file__).read_bytes()).hexdigest(),
                  script=args.script, arguments=args.arguments)
    original_init, original_call = ModelWeights.__init__, Linear.__call__

    def prepare(model, path, max_context, quantization=None):
        config = decoder_config(json.loads((Path(path)/'config.json').read_text()))
        geometry = dict(model_type='qwen2', hidden_size=896, num_hidden_layers=24,
                        num_attention_heads=14, num_key_value_heads=2, intermediate_size=4864,
                        vocab_size=151936, head_dim=64)
        if (any(config.get(key) != value for key,value in geometry.items())
                or not 1 <= max_context <= 2048 or quantization not in (None, 'q4_0')):
            raise ValueError('experiment requires Qwen2.5-0.5B Q4, context <=2048')
        # Only host preparation bypasses the existing Metal admission guard.
        # Restore device selection before constructing any GPU executor.
        with patch('damendaemon.qwen2.selected_device', return_value='cuda'):
            original_init(model, path, max_context, quantization)

    def dispatch(owner, x, weight, bias=None):
        if x.device.type == 'mps' and weight.dtype == torch.uint8:
            report['selected_calls'] += 1
            return quantized_metal_q4.linear(x, weight, owner.scales[id(weight)], bias)
        report['fallback_calls'] += 1
        return original_call(owner, x, weight, bias)

    original_argv = sys.argv
    try:
        sys.argv = [args.script, *args.arguments]
        with patch.object(ModelWeights, '__init__', prepare), patch.object(Linear, '__call__', dispatch):
            runpy.run_path(args.script, run_name='__main__')
        report['completed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        sys.argv = original_argv
        Path(args.report).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':main()

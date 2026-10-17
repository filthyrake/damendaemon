"""Real-device chunk boundaries, slot isolation and abandoned-prefix reuse.

This complements independent model-reference replays; the full-prompt control
here uses damendaemon itself and is not independent model correctness or speed proof.
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
from pathlib import Path
import torch
from damendaemon.qwen2 import ModelWeights
from damendaemon.runtime import require_gpu, make_executor
from damendaemon.device import device_name, selected_accelerator


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    model = ModelWeights(args.model, 512)
    executor = make_executor(model, max_batch=3)
    if not getattr(executor, 'supports_chunked_prefill', False):
        raise ValueError('requires the supported hybrid chunk path')
    corpus = model.raw_tokens('The parcel is in cabinet seven, drawer three. ' * 100)
    checks = []
    report = dict(passed=False, device=device_name(), accelerator=selected_accelerator(),
                  model=args.model, context=512, quantization=model.quantization,
                  reference='damendaemon full-prompt control; independent replays are separate', checks=checks)

    def compare(actual, expected, label):
        actual, expected = actual.float().cpu(), expected.float().cpu()
        delta = actual - expected
        rms = (delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6)).item()
        absolute = delta.abs().max().item()
        greedy = actual.argmax().item() == expected.argmax().item()
        passed = bool(torch.isfinite(actual).all()) and greedy and rms < .01 and absolute < .5
        checks.append(dict(label=label, relative_rms=rms, max_absolute=absolute,
                           greedy_match=greedy, passed=passed))
        assert passed, label

    try:
        anchor = corpus[32:63]
        baseline = executor.prefill(anchor, 1)
        anchor_tokens, anchor_logits = [], []
        for _ in range(16):
            token = baseline.argmax().item()
            anchor_tokens.append(token)
            baseline, _ = executor.decode_batch([token], [1])
            anchor_logits.append(baseline.float().cpu().clone())
        for length in (1, 2, 3, 4, 7, 31, 65, 257):
            prompt = corpus[:length]
            expected = executor.prefill(prompt, 0).float().cpu().clone()
            executor.prefill(anchor, 1)
            start, step = 0, 0
            while start < length:
                width = (1, 2, 3, 5, 17, 127)[step % 6]
                end = min(start + width, length)
                output = executor.prefill_chunk(prompt[start:end], 2, start, final=end == length)
                if end < length:
                    assert output is None, 'partial prompt emitted logits'
                else:
                    compare(output, expected, f'chunked-prefix-{length}')
                active, _ = executor.decode_batch([anchor_tokens[step]], [1])
                compare(active, anchor_logits[step], f'isolated-decode-{length}-{step}')
                start, step = end, step + 1
            # A cancelled partial request leaves state, which a new offset-zero
            # request must reset even when its first chunk is shorter than convolution width.
            executor.prefill_chunk(corpus[5:8], 2, 0, final=False)
            output = executor.prefill_chunk(prompt[:1], 2, 0, final=length == 1)
            if length > 1:
                output = executor.prefill_chunk(prompt[1:], 2, 1, final=True)
            compare(output, expected, f'abandoned-prefix-reuse-{length}')
            before = [t.clone() for t in (*executor.conv.values(), *executor.state.values(), *executor.kv.values())]
            lengths = list(executor.lengths)
            try:
                executor.prefill_chunk([corpus[0]], 2, length + 1)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid offset was accepted')
            assert lengths == executor.lengths
            assert all(torch.equal(a, b) for a, b in zip(before,
                (*executor.conv.values(), *executor.state.values(), *executor.kv.values())))
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

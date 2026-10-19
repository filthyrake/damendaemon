"""Real-checkpoint Q8_0 KV-cache drift study on the hybrid executor (#535 gate 1).

Runs the identical pinned checkpoint twice in one process — default KV cache
(FP16 compute, FP16 cache) vs DAMENDAEMON_HYBRID_KV=q8_0 — over a fixed
teacher-forced corpus, recording perplexity, next-token top-1 agreement,
per-text relative logit RMS against the baseline run, and a 32-step greedy
continuation per mode. The #23 drift-study pattern: capability evidence, not a
quality benchmark; no performance claims.

GPU execution requires the shared-CUDA guard wrapper (guard_shared_cuda).
"""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from scripts.experiments.split_gguf import save

TEXTS = [
    'The Earth travels around the Sun once every year. The Moon travels around the Earth.',
    'A binary search repeatedly divides a sorted list in half until it finds the requested value.',
    'Water freezes at zero degrees Celsius under ordinary atmospheric pressure.',
    'A gardener planted tomato seeds in spring and watered them every morning.',
    'The sum of seven and five is twelve. Multiplying three by four also gives twelve.',
    'In Python, a dictionary maps keys to values, while a list stores an ordered sequence of elements.',
    'The train left the station at noon and arrived in the next city two hours later.',
    'Bonjour, comment allez-vous? Je voudrais un café et un verre d’eau, s’il vous plaît.',
    '東京は日本の首都です。今日は空が青く、穏やかな風が吹いています。',
    'A reliable program validates its inputs, handles errors, and keeps unrelated requests isolated.',
    'The network is built from layers, and each layer learns to represent the input a little more '
    'abstractly than the last. Early layers detect simple patterns such as edges or frequent word '
    'pairs, while deeper layers compose those patterns into phrases, sentences, and arguments. '
    'Attention lets every position read from every other position, so a pronoun can find the noun '
    'it refers to even when many words separate them. Training adjusts millions of weights a tiny '
    'step at a time until the predictions on the next word stop improving.',
    'A cache stores results that are expensive to compute so they can be reused later. The trick is '
    'knowing when the stored value is still valid. If the underlying data changes, every cached copy '
    'derived from it must be discarded, or the system will quietly return answers that were true '
    'yesterday and wrong today. Engineers call this invalidation, and it is one of the two genuinely '
    'hard problems of computing, along with naming things. A cache that is never invalidated is not '
    'a cache at all but a very determined archive of mistakes.',
    'Quantization maps a wide range of numbers onto a narrow grid of representable steps. Storing a '
    'value as an eight-bit integer with one shared scale per group of channels costs a quarter of '
    'the memory of a thirty-two-bit float, at the price of rounding every value to the nearest step. '
    'If the values in a group are all similar in magnitude, the steps are fine and the error is '
    'small; if one value is enormous and the rest are tiny, the scale must stretch to fit the '
    'outlier and the small values collapse onto zero. Grouping exists to keep that from happening '
    'too often.',
    'The decay of attention over long contexts is not a flaw but a design choice. A model that '
    'weighted every previous token equally would spend most of its attention budget on text that '
    'cannot possibly matter to the next word. Recency bias, learned rather than hard-coded, lets '
    'the model decide for itself how far back to look. The measured drift between a full-precision '
    'key-value cache and a compressed one is therefore largest exactly where attention reaches '
    'furthest back, and smallest for the tokens the model was going to ignore anyway.',
]
GREEDY_PROMPT = ('Quantization trades exact arithmetic for memory. The measured '
                 'difference between the two caches appears')
GREEDY_STEPS = 32
MODES = ('none', 'q8_0')


def corpus(tokenizer, context):
    # HF `tokenizers` Tokenizer: encode returns an Encoding with .ids.
    limit = context - GREEDY_STEPS - 1
    return [tokenizer.encode(text).ids[:limit] for text in TEXTS]


@torch.inference_mode()
def run_mode(mode, model_path, context):
    os.environ['DAMENDAEMON_HYBRID_KV'] = mode
    from damendaemon.qwen2 import ModelWeights
    from damendaemon.hybrid import HybridExecutor
    model = ModelWeights(model_path, max_context=context, quantization='none')
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    rows = []
    for ids in corpus(model.tokenizer, context):
        token_ids = torch.tensor([ids], device=executor.device)
        positions = torch.arange(len(ids), device=executor.device)[None]
        hidden = executor.forward_layers(executor.embedding(token_ids), positions,
                                         torch.tensor([0], device=executor.device), True)
        logits = executor.output_logits(hidden)[0].float().cpu()
        # Teacher forcing: logits[t] predicts token t+1.
        predictions, targets = logits[:-1], torch.tensor(ids[1:])
        nll = torch.nn.functional.cross_entropy(predictions, targets, reduction='sum').item()
        top1 = int((predictions.argmax(-1) == targets).sum().item())
        rows.append(dict(tokens=len(ids), nll=nll, top1=top1, logits=logits))
    prompt_ids = model.tokenizer.encode(GREEDY_PROMPT).ids
    prompt_ids = list(prompt_ids)[:context - GREEDY_STEPS]
    continuation = list(prompt_ids)
    executor.prefill(continuation)
    for _ in range(GREEDY_STEPS):
        logits, next_id = executor.decode(continuation[-1])
        continuation.append(int(next_id[0]))
    greedy = model.tokenizer.decode(continuation[len(prompt_ids):])
    executor.close()
    return dict(rows=rows, greedy=greedy)


def summarize(mode_output, baseline_rows=None):
    rows = mode_output['rows']
    tokens = sum(row['tokens'] - 1 for row in rows)
    nll = sum(row['nll'] for row in rows)
    top1 = sum(row['top1'] for row in rows)
    summary = dict(tokens=tokens, perplexity=round(math.exp(nll/tokens), 4),
                   top1_agreement=round(top1/tokens, 4), greedy=mode_output['greedy'])
    if baseline_rows is not None:
        relative = []
        for row, base in zip(rows, baseline_rows):
            difference = row['logits'] - base['logits']
            relative.append(round((difference.norm()/base['logits'].norm()).item(), 6))
        summary['per_text_relative_logit_rms'] = relative
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    report: dict = dict(completed=False, model=str(args.model), context=args.context,
                        torch=torch.__version__,
                        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                        modes={})
    save(args.output, report)
    outputs = {}
    for mode in MODES:
        outputs[mode] = run_mode(mode, args.model, args.context)
        report['modes'][mode] = summarize(outputs[mode])
        save(args.output, report)
    report['modes']['q8_0']['per_text_relative_logit_rms'] = \
        summarize(outputs['q8_0'], outputs['none']['rows'])['per_text_relative_logit_rms']
    report['greedy_identical'] = outputs['none']['greedy'] == outputs['q8_0']['greedy']
    report['kv_bytes_note'] = ('baseline cache is FP16 (compute dtype); q8_0 stores int8 '
                               'plus per-group FP16 scales and dequantizes on read')
    report['completed'] = True
    save(args.output, report)
    print(json.dumps({key: report[key] for key in ('modes', 'greedy_identical')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

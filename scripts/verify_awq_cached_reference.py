"""Replay content-pinned independent AWQ oracle logits against a runtime backend."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from dllm.awq_model import AwqWeights, config
from dllm.runtime import make_executor, require_gpu
from dllm.models import load
from scripts.awq_reference_cache import fingerprint


def load_oracle(cache, reference_report, model):
    cache_path = Path(cache)
    with cache_path.open('rb') as stream:
        cache_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    origin = json.loads(Path(reference_report).read_text())
    if origin.get('reference_cache', {}).get('sha256') != cache_hash:
        raise ValueError('oracle cache differs from its independent creation report')
    with safe_open(str(cache_path), framework='pt', device='cpu') as reader:
        metadata = json.loads(reader.metadata()['qualification'])
        expected_keys = {f'case_{i}_step_{j}' for i in range(8) for j in range(8)}
        if (metadata.get('version') != 1 or set(metadata.get('cases', [])) != set(range(8))
                or len(metadata.get('prompts', [])) != 8
                or metadata.get('steps') != 8 or set(reader.keys()) != expected_keys
                or metadata.get('reference_dtype') != 'float32'
                or metadata.get('teacher_forcing') != 'independent reference greedy tokens'):
            raise ValueError('expected a complete independent FP32 oracle')
        if metadata['checkpoint'] != fingerprint(model):
            raise ValueError('checkpoint or tokenizer differs from the cached oracle')
        # Bind tensor files to the hashes computed before oracle execution too.
        for name, digest in origin['checkpoint_sha256'].items():
            if metadata['checkpoint'].get(name) != digest:
                raise ValueError('checkpoint changed during oracle creation')
        targets = {name: reader.get_tensor(name) for name in reader.keys()}
    vocab = config(model)['vocab_size']
    if any(not isinstance(ids, list) or not ids or any(type(token) is not int or not 0 <= token < vocab for token in ids)
           for ids in metadata['prompts']):
        raise ValueError('invalid oracle input tokens')
    if any(t.dtype != torch.float32 or tuple(t.shape) != (1, vocab)
           or not bool(torch.isfinite(t).all()) for t in targets.values()):
        raise ValueError('invalid oracle logit tensor')
    return metadata, targets, cache_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--reference-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experimental-awq-marlin', metavar='LIBRARY')
    args = parser.parse_args()
    metadata, targets, cache_hash = load_oracle(args.cache, args.reference_report, args.model)
    require_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    experiment = None
    if args.experimental_awq_marlin:
        from experiments.awq_marlin_dispatch import install_validation_dispatch
        experiment = install_validation_dispatch(args.experimental_awq_marlin)
    source = load(args.model, metadata['context'], 'none')
    if not isinstance(source, AwqWeights):
        raise ValueError('cached AWQ qualification requires the ordinary AWQ factory')
    report = dict(passed=False, experiment=experiment, context=metadata['context'],
                  oracle_sha256=cache_hash, oracle=metadata, cases=[],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        with torch.inference_mode():
            for graph in (False, True):
                executor = make_executor(source, use_graph=graph)
                report['backend'] = executor.kernel_backend
                report['precision'] = dict(matrix=str(executor.compute_dtype), residual=str(executor.residual_dtype))
                tensors = [*executor.w.values(), *executor.quant_linear.scales.values(),
                           *(m.zeros for m in executor.quant_linear.awq.values())]
                actual_bytes = sum(t.numel()*t.element_size() for t in {id(t): t for t in tensors}.values())
                if actual_bytes != source.bytes:
                    raise AssertionError('GPU storage disagrees with packed checkpoint admission')
                report['gpu_weight_bytes'] = actual_bytes
                for index, ids in enumerate(metadata['prompts']):
                    actual = executor.prefill(ids)
                    case = dict(graph=graph, case=index, prompt_tokens=len(ids), checks=[])
                    report['cases'].append(case)
                    for step in range(8):
                        target = targets[f'case_{index}_step_{step}'].cuda()
                        error = actual.float()-target
                        rms = (error.square().mean()/target.square().mean().clamp_min(1e-12)).sqrt().item()
                        maximum = error.abs().max().item()
                        token, observed = target.argmax(-1).item(), actual.argmax(-1).item()
                        passed = bool(torch.isfinite(actual).all()) and rms < .01 and maximum < .5 and token == observed
                        case['checks'].append(dict(step=step, relative_rms=rms, max_absolute=maximum,
                                                   reference_token=token, actual_token=observed, passed=passed))
                        if step < 7:
                            actual, next_tokens = executor.decode(token)
                            del next_tokens
                    print(json.dumps(dict(graph=graph, case=index,
                        passed=all(check['passed'] for check in case['checks']))), flush=True)
                del executor, tensors, actual
                gc.collect()
                torch.cuda.empty_cache()
        report['passed'] = all(check['passed'] for case in report['cases'] for check in case['checks'])
        if not report['passed']:
            raise AssertionError('cached independent AWQ logit gate failed')
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

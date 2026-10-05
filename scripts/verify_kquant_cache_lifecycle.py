"""Prove the experimental resident projection cache releases model residency."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import weakref

import torch

from dllm.gguf_model import GgufWeights
from dllm.runtime import make_executor, require_gpu
from experiments import kquant_dense_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--expanded-control', action='store_true',
                        help='exercise normal dense execution with independently expanded weights, no cache hook')
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    report = dict(experiment=None if args.expanded_control else kquant_dense_cache.install_validation_dispatch(),
                  expanded_control=args.expanded_control,
                  passed=False, cycles=[], device=torch.cuda.get_device_name())
    with open(args.model, 'rb') as handle:
        report['model_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    source = GgufWeights(args.model, 2048, 'none')
    if args.expanded_control:
        from verify_kquant_runtime import expanded
        source = expanded(source)
    prompt = source.raw_tokens('Explain how the ocean influences coastal weather.')
    previous_released = None
    try:
        with torch.inference_mode():
            for cap in ((1, 1, 1) if args.expanded_control else (3 * 1024**3, 3 * 1024**3, 1)):
                kquant_dense_cache.CACHE_LIMIT = cap
                gc.collect()
                torch.cuda.empty_cache()
                before = torch.cuda.memory_allocated()
                executor = make_executor(source, use_graph=True, max_batch=4)
                cached = getattr(executor.quant_linear, 'experimental_dense_cache_bytes', 0)
                refs = [weakref.ref(executor)] + [weakref.ref(value) for value in executor.w.values()]
                refs += [weakref.ref(value) for value in getattr(
                    executor.quant_linear, 'experimental_dense_cache', {}).values()]
                assert (cached > 0) if cap > 1 else (cached == 0)
                logits = executor.prefill_batch([prompt] * 4, list(range(4)))
                logits, tokens = executor.decode_batch(logits.argmax(-1).tolist(), list(range(4)))
                assert torch.isfinite(logits).all()
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                del logits, tokens, executor
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                released = torch.cuda.memory_allocated()
                item = dict(cap=cap, cached_bytes=cached, before=before, resident=resident,
                            after_release=released, references_released=all(ref() is None for ref in refs))
                report['cycles'].append(item)
                assert item['references_released'], 'cache retains its executor or tensors'
                assert resident - released >= cached, 'cache allocation remains resident'
                if previous_released is not None:
                    assert released <= previous_released + 1024**2, 'residency grows after repeated unload'
                previous_released = released
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

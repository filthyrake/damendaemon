"""Bounded real-model projection probe; full-model/HTTP gates remain separate."""
import argparse
import gc
import json
import statistics
import torch

from dllm.cuda_warmup import warmup_stream
from dllm.qwen2 import ModelWeights
from dllm.quantization import dequantize
from dllm.runtime import require_gpu
from scripts.experiments.hip_q8_small_batch import candidate, ordinary


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(73)
    model = ModelWeights(args.model, 4096, quantization='q8_0')
    if model.config['model_type'] != 'qwen3_5_text':
        raise ValueError('use the modern Qwen3.5 target')
    representatives = {}
    for name, weight in model.weights.items():
        if weight.dtype == torch.int8 and name != 'head':
            representatives.setdefault(tuple(weight.shape), name)
    report = dict(passed=False, model_path=str(model.path), device=torch.cuda.get_device_name(0),
                  scope='projection microbenchmark, not model or serving performance', cases=[])
    try:
        for (n, k), name in representatives.items():
            host_weight, host_scales = model.weights[name], model.scales[name]
            reference_weight = dequantize(host_weight, host_scales, torch.float16).float()
            weight, scales = host_weight.cuda(), host_scales.cuda()
            for rows in (2, 3, 4):
                host_x = torch.randn(rows, k).half()
                reference = torch.nn.functional.linear(host_x.float(), reference_weight)
                x = host_x.cuda()
                graphs, values = {}, {}
                case = dict(name=name, rows=rows, inputs=k, outputs=n, checks={}, timings={})
                for label, fn in (('ordinary', ordinary), ('candidate', candidate)):
                    with warmup_stream():
                        for _ in range(3):
                            fn(x, weight, scales)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        value = fn(x, weight, scales)
                    graphs[label], values[label] = graph, value
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = value.float().cpu()
                    error = actual-reference
                    relative = (error.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1.e-6)).item()
                    absolute = error.abs().max().item()
                    case['checks'][label] = dict(relative_rms=relative, max_absolute=absolute,
                        passed=bool(torch.isfinite(actual).all()) and relative < .001 and absolute < .01)
                    case['timings'][label] = []
                for _ in range(10):
                    for label in ('ordinary', 'candidate', 'candidate', 'ordinary'):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):
                            graphs[label].replay()
                        end.record()
                        end.synchronize()
                        case['timings'][label].append(start.elapsed_time(end)*10)  # us/replay
                case['median_us'] = {label: statistics.median(v) for label, v in case['timings'].items()}
                report['cases'].append(case)
                del graphs, values, graph, value, x
                gc.collect()
            del weight, scales, reference_weight
            gc.collect()
            torch.cuda.empty_cache()
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['passed'] = all(c['checks'][label]['passed'] for c in report['cases'] for label in ('ordinary', 'candidate'))
        if not report['passed']:
            raise AssertionError('projection arithmetic failed its fixed gate')
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        with open(args.output, 'w') as file:
            json.dump(report, file, indent=2)
            file.write('\n')


if __name__ == '__main__':
    main()

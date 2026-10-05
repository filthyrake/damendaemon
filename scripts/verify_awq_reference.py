"""Independent AWQ checkpoint logits: upstream coefficient decoder plus HF math."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from dllm.awq_model import AwqWeights
from dllm.runtime import make_executor, require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--upstream-source', required=True)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.context < 512:
        parser.error('context must be at least 512')
    require_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(args.model)
    source = AwqWeights(path, args.context, 'none')
    report = dict(passed=False, context=args.context, cases=[], model=str(path.resolve()),
                  host_bytes=source.bytes, torch=torch.__version__, device=torch.cuda.get_device_name(),
                  reference='pinned upstream AWQ coefficient decoder; independent HF eager FP16 model execution',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  loader_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'dllm/awq_model.py').read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'dllm/awq_kernels.py').read_bytes()).hexdigest())
    with (path/'model.safetensors').open('rb') as stream:
        report['model_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        upstream = Path(args.upstream_source).read_text()
        dependency = 'from vllm.triton_utils import tl, triton'
        if upstream.count(dependency) != 1:
            raise ValueError('unexpected upstream validation source')
        report['upstream_source_sha256'] = hashlib.sha256(upstream.encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='dllm-awq-reference-') as temporary:
            reference_path = Path(temporary)/'awq_reference.py'
            reference_path.write_text(upstream.replace(dependency, 'import triton\nimport triton.language as tl'))
            spec = importlib.util.spec_from_file_location('dllm_awq_model_reference', reference_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            state = {}
            with torch.inference_mode(), safe_open(str(path/'model.safetensors'), framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    if name.endswith(('.qzeros', '.scales')):
                        continue
                    if name.endswith('.qweight'):
                        prefix = name.rsplit('.', 1)[0]
                        weight, scales, zeros = (reader.get_tensor(prefix+suffix).cuda()
                                                  for suffix in ('.qweight', '.scales', '.qzeros'))
                        decoded = module.awq_dequantize_triton(weight, scales, zeros)
                        state[prefix+'.weight'] = decoded.T.contiguous().cpu()
                        del weight, scales, zeros, decoded
                    else:
                        state[name] = reader.get_tensor(name)
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        del config.quantization_config
        config.tie_word_embeddings = False
        with no_init_weights():
            reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='eager')
        reference.load_state_dict(state, strict=True, assign=True)
        del state
        reference.eval().to('cuda', dtype=torch.float16)
        prompts = [source.chat_tokens(messages) for messages in (
            [{'role': 'user', 'content': 'What is 2 + 2? Answer with one number.'}],
            [{'role': 'user', 'content': 'Remember the code ZX-914.'},
             {'role': 'assistant', 'content': 'The code is ZX-914.'},
             {'role': 'user', 'content': 'Repeat only the code.'}],
            [{'role': 'user', 'content': 'Translate into Spanish: The café opens at eight. 👋'}],
            [{'role': 'user', 'content': 'Write a Python function that returns the largest number in a list.'}])]
        corpus = source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
        for length in (16, 64, 129, args.context-8):
            prompts.append((corpus*((length+len(corpus)-1)//len(corpus)))[:length])
        report['prompts'] = prompts
        with torch.inference_mode():
            for graph in (False, True):
                executor = make_executor(source, use_graph=graph)
                if executor.kernel_backend != 'triton+awq-gemm':
                    raise AssertionError(f'AWQ backend was not selected: {executor.kernel_backend}')
                gpu_tensors = [*executor.w.values(), *executor.quant_linear.scales.values(),
                               *(m.zeros for m in executor.quant_linear.awq.values())]
                actual_bytes = sum(t.numel()*t.element_size() for t in {id(t): t for t in gpu_tensors}.values())
                if actual_bytes != source.bytes:
                    raise AssertionError('GPU packed weight storage disagrees with host admission')
                report['gpu_weight_bytes'] = actual_bytes
                report['workspace_reserve_bytes'] = executor.packed_workspace_bytes
                report['precision'] = dict(matrix=str(executor.compute_dtype), residual=str(executor.residual_dtype))
                for index, ids in enumerate(prompts):
                    expected = reference(torch.tensor([ids], device='cuda'), use_cache=True)
                    actual = executor.prefill(ids)
                    case = dict(graph=graph, case=index, prompt_tokens=len(ids), checks=[])
                    report['cases'].append(case)
                    for step in range(8):
                        target = expected.logits[:, -1, :]
                        error = actual.float()-target.float()
                        rms = (error.square().mean()/target.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        maximum = error.abs().max().item()
                        token, observed = target.argmax(-1).item(), actual.argmax(-1).item()
                        best = target.float().topk(2, dim=-1).values
                        passed = bool(torch.isfinite(actual).all()) and rms < .01 and maximum < .5 and token == observed
                        case['checks'].append(dict(step=step, relative_rms=rms, max_absolute=maximum,
                                                   reference_token=token, actual_token=observed,
                                                   reference_top_two_margin=(best[0, 0]-best[0, 1]).item(), passed=passed))
                        if step < 7:
                            expected = reference(torch.tensor([[token]], device='cuda'),
                                                 past_key_values=expected.past_key_values, use_cache=True)
                            actual = executor.decode(token)
                    print(json.dumps(dict(graph=graph, case=index, prompt_tokens=len(ids),
                                          passed=all(c['passed'] for c in case['checks']))), flush=True)
                del executor, actual, expected, gpu_tensors
                gc.collect()
                torch.cuda.empty_cache()
        report['passed'] = all(check['passed'] for case in report['cases'] for check in case['checks'])
        if not report['passed']:
            raise AssertionError('independent AWQ model logit gate failed')
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

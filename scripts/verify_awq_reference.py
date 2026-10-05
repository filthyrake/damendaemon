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


def install_rounded_attention_diagnostic():
    """Isolate eager FP16 attention rounding; never installed by the runtime."""
    from dllm.qwen2 import Executor

    def attention(self, query, key, value, positions):
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        scores = (query @ key.transpose(-1, -2)) * (self.dim ** -.5)
        allowed = torch.arange(key.shape[-2], device=query.device) <= positions[..., None]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        return probabilities @ value

    def prefill(self, query, key, value, query_start=0):
        positions = torch.arange(query_start, query_start+query.shape[-2], device=query.device)
        return attention(self, query, key, value, positions)

    def decode(self, query, key, value, positions, slots=None):
        if slots is not None:
            key, value = key.index_select(0, slots), value.index_select(0, slots)
        return attention(self, query, key, value, positions[:, None, None])

    Executor.prefill_attention = prefill
    Executor.decode_attention = decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--upstream-source', required=True)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-attention', choices=('eager', 'sdpa'), default='eager',
                        help='independent HF attention implementation; recorded as qualification scope')
    parser.add_argument('--experimental-awq-decode', action='store_true')
    parser.add_argument('--experimental-awq-marlin', metavar='LIBRARY')
    parser.add_argument('--require-awq-decode', action='store_true', help='assert ordinary A2 decode selection')
    parser.add_argument('--diagnostic-expanded-products', action='store_true',
                        help='isolate packed GEMM errors using temporary expanded FP16 products')
    parser.add_argument('--diagnostic-separate-products', action='store_true',
                        help='use original Q/K/V and gate/up matrix shapes for expanded products')
    parser.add_argument('--diagnostic-rounded-attention', action='store_true',
                        help='isolate eager FP16 attention rounding from fused attention')
    parser.add_argument('--diagnostic-layer-trace', action='store_true',
                        help='record first-prompt residual error at each decoder layer')
    args = parser.parse_args()
    if args.context < 512:
        parser.error('context must be at least 512')
    require_gpu()
    experiment = None
    if args.experimental_awq_marlin:
        if args.experimental_awq_decode or args.diagnostic_expanded_products:
            parser.error('Marlin cannot be combined with other product dispatch overrides')
        from experiments.awq_marlin_dispatch import install_validation_dispatch
        experiment = install_validation_dispatch(args.experimental_awq_marlin)
    if args.experimental_awq_decode:
        from experiments.awq_decode import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.diagnostic_rounded_attention:
        install_rounded_attention_diagnostic()
    if args.diagnostic_expanded_products:
        from dllm import awq_kernels
        awq_kernels.linear = lambda x, matrix, bias=None: torch.nn.functional.linear(
            x, awq_kernels.expand(matrix), bias)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(args.model)
    source = AwqWeights(path, args.context, 'none')
    if args.require_awq_decode:
        from dllm.awq_decode_policy import use_small_rows
        matrix = source.awq_matrices['0.qkv.weight']
        if not use_small_rows(1, matrix.inputs, matrix.outputs, torch.device('cuda:0')):
            raise AssertionError('normal AWQ decode policy was not selected')
    if args.diagnostic_separate_products:
        if not args.diagnostic_expanded_products:
            parser.error('separate products requires diagnostic expanded products')
        from dllm import awq_kernels
        c = source.config
        q, kv = c['num_attention_heads']*c['head_dim'], c['num_key_value_heads']*c['head_dim']
        def separate_products(x, matrix, bias=None):
            weight = awq_kernels.expand(matrix)
            sizes = ([q, kv, kv] if matrix.inputs == c['hidden_size'] and matrix.outputs == q+2*kv else
                     [c['intermediate_size']]*2 if matrix.inputs == c['hidden_size'] and matrix.outputs == 2*c['intermediate_size'] else
                     [matrix.outputs])
            biases = [None]*len(sizes) if bias is None else bias.split(sizes)
            return torch.cat([torch.nn.functional.linear(x, w.contiguous(), b)
                              for w, b in zip(weight.split(sizes), biases)], -1)
        awq_kernels.linear = separate_products
    report = dict(passed=False, diagnostic_expanded_products=args.diagnostic_expanded_products,
                  experiment=experiment,
                  require_awq_decode=args.require_awq_decode,
                  diagnostic_rounded_attention=args.diagnostic_rounded_attention,
                  diagnostic_separate_products=args.diagnostic_separate_products,
                  context=args.context, cases=[], model=str(path.resolve()),
                  host_bytes=source.bytes, torch=torch.__version__, device=torch.cuda.get_device_name(),
                  reference='pinned upstream AWQ coefficient decoder; independent HF FP16 model execution',
                  reference_attention=args.reference_attention,
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
            with torch.no_grad(), safe_open(str(path/'model.safetensors'), framework='pt', device='cpu') as reader:
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
            reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation=args.reference_attention)
        reference.requires_grad_(False)
        reference.load_state_dict(state, strict=True, assign=True)
        del state
        # Parameters are already FP16. Casting the whole module also truncates
        # HF's intentionally FP32 rotary frequencies, corrupting the oracle.
        reference.eval().to('cuda')
        if reference.model.rotary_emb.inv_freq.dtype != torch.float32:
            raise AssertionError('reference rotary frequencies must retain FP32')
        report['reference_rotary_dtype'] = str(reference.model.rotary_emb.inv_freq.dtype)
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
                    handles, reference_layers, actual_layers = [], [], []
                    reference_parts, actual_parts = {}, {}
                    if args.diagnostic_layer_trace and not graph and index == 0:
                        def record_reference(module, inputs):
                            reference_layers.append(inputs[0].detach().cpu())
                        for layer in [*reference.model.layers, reference.model.norm]:
                            handles.append(layer.register_forward_pre_hook(record_reference))
                        first = reference.model.layers[0]
                        for name, module in [('norm1', first.input_layernorm), ('norm2', first.post_attention_layernorm),
                                             ('q', first.self_attn.q_proj), ('k', first.self_attn.k_proj),
                                             ('v', first.self_attn.v_proj), ('out', first.self_attn.o_proj),
                                             ('gate', first.mlp.gate_proj), ('up', first.mlp.up_proj), ('down', first.mlp.down_proj)]:
                            def record_part(module, inputs, output, name=name):
                                reference_parts[name] = output.detach().cpu().squeeze(0)
                            handles.append(module.register_forward_hook(record_part))
                    expected = reference(torch.tensor([ids], device='cuda'), use_cache=True)
                    for handle in handles:
                        handle.remove()
                    original_norm = executor.norm
                    original_linear = executor.linear
                    original_attention = executor.prefill_attention
                    if handles:
                        def record_norm(x, weight):
                            actual_layers.append(x.detach().cpu())
                            result = original_norm(x, weight)
                            if len(actual_layers) <= 2:
                                actual_parts['norm'+str(len(actual_layers))] = result.detach().cpu()
                            return result
                        def record_linear(x, weight, bias=None):
                            result = original_linear(x, weight, bias)
                            for name in ('qkv.weight', 'out', 'gate_up', 'down'):
                                if weight is executor.w['0.'+name]:
                                    actual_parts[name] = result.detach().cpu()
                            return result
                        executor.norm = record_norm
                        executor.linear = record_linear
                        attention_calls = 0
                        def record_attention(query, key, value, query_start=0):
                            nonlocal attention_calls
                            result = original_attention(query, key, value, query_start)
                            if attention_calls == 0:
                                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
                                rq, rk, rv = [reference_parts[name].cuda().view(1, len(ids), -1, executor.dim).transpose(1, 2)
                                              for name in ('q', 'k', 'v')]
                                positions = torch.arange(len(ids), device='cuda')[None, :]
                                cos, sin = reference.model.rotary_emb(rq, positions)
                                rq, rk = apply_rotary_pos_emb(rq, rk, cos, sin)
                                scores = (rq @ repeat_kv(rk, executor.heads//executor.kv_heads).transpose(-1, -2)) * executor.dim**-.5
                                mask = positions[:, :, None] >= positions[:, None, :]
                                scores = scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min)
                                ra = scores.softmax(-1, dtype=torch.float32).half() @ repeat_kv(rv, executor.heads//executor.kv_heads)
                                report['first_attention_parts'] = []
                                for name, observed, target in [('query', query, rq), ('key', key, rk), ('value', value, rv), ('attention', result, ra)]:
                                    delta = observed.float()-target.float()
                                    report['first_attention_parts'].append(dict(name=name,
                                        relative_rms=(delta.square().mean()/target.float().square().mean().clamp_min(1e-12)).sqrt().item(),
                                        max_absolute=delta.abs().max().item()))
                            attention_calls += 1
                            return result
                        executor.prefill_attention = record_attention
                    actual = executor.prefill(ids)
                    executor.norm = original_norm
                    executor.linear = original_linear
                    executor.prefill_attention = original_attention
                    if handles:
                        report['layer_trace'] = []
                        # Two norms per layer; the final norm sees only the last token.
                        for layer, (observed, target) in enumerate(zip(actual_layers[::2], reference_layers)):
                            delta = observed[-1].float()-target[0, -1].float()
                            report['layer_trace'].append(dict(layer=layer,
                                last_token_relative_rms=(delta.square().mean()/target[0, -1].float().square().mean().clamp_min(1e-12)).sqrt().item(),
                                last_token_max_absolute=delta.abs().max().item()))
                        reference_parts['qkv.weight'] = torch.cat([reference_parts[k] for k in ('q', 'k', 'v')], -1)
                        reference_parts['gate_up'] = torch.cat([reference_parts[k] for k in ('gate', 'up')], -1)
                        report['first_layer_parts'] = []
                        for name, observed in actual_parts.items():
                            target = reference_parts[name]
                            delta = observed.float()-target.float()
                            report['first_layer_parts'].append(dict(name=name,
                                relative_rms=(delta.square().mean()/target.float().square().mean().clamp_min(1e-12)).sqrt().item(),
                                max_absolute=delta.abs().max().item()))
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
                            actual, _ = executor.decode(token)
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

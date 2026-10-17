"""Isolate Qwen2 residual/nonlinear/rotary precision on a fixed GGUF trace.

Diagnostic only: changes executor fields after constructing an eager executor.
No production policy is enabled by this experiment.
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights
from verify_gguf_reference import reference_weights
from damendaemon.gguf_model import GgufWeights
from damendaemon.runtime import make_executor
from damendaemon.rotary import inverse_frequencies, attention_factor


def main():
    p = argparse.ArgumentParser(); p.add_argument('gguf'); p.add_argument('hf_config'); p.add_argument('teacher_trace'); p.add_argument('--output', required=True)
    p.add_argument('--rotary-controls', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    source = GgufWeights(args.gguf, 512, 'none')
    assert source.config['model_type'] == 'qwen2'
    config = AutoConfig.from_pretrained(args.hf_config, local_files_only=True); config.tie_word_embeddings = False
    with no_init_weights(): reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='eager')
    state, _, _ = reference_weights(args.gguf); reference.load_state_dict(state, strict=True, assign=True); del state
    reference.eval().to(device='cuda', dtype=torch.float32)
    teacher = json.loads(Path(args.teacher_trace).read_text())
    prompts = [source.chat_tokens([{'role':'user','content':text}]) for text in
               ('What is 2 + 2? Answer with one number.', 'Reply in French: Hello, how are you?')]
    base = source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
    for length in (129, 505): prompts.append((base * ((length+len(base)-1)//len(base)))[:length])
    traces = []
    with torch.inference_mode():
        for index, ids in enumerate(prompts):
            forced = next(c['reference_tokens'] for c in teacher['cases'] if not c['graph'] and c['case'] == index)
            expected = reference(torch.tensor([ids], device='cuda'), use_cache=True)
            logits = []
            for step, token in enumerate(forced):
                logits.append(expected.logits[:, -1].cpu())
                if step < 7: expected = reference(torch.tensor([[token]], device='cuda'), past_key_values=expected.past_key_values, use_cache=True)
            traces.append((forced, logits))
        del reference, expected
        gc.collect(); torch.cuda.empty_cache()
        executor = make_executor(source, use_graph=False)
        original_cos, original_sin = executor.cos, executor.sin
        frequencies = inverse_frequencies(executor.c, 'cuda')
        angles = torch.outer(torch.arange(512, device='cuda', dtype=torch.float32), frequencies)
        angles = torch.cat((angles, angles), dim=-1)
        scale = attention_factor(executor.c)
        precise_cos, precise_sin = angles.cos()*scale, angles.sin()*scale
        results = []
        policies = (('baseline', False, False, False), ('residual', True, False, False),
                    ('nonlinear', False, True, False), ('residual_nonlinear', True, True, False),
                    ('mixed_all', True, True, True))
        if args.rotary_controls:
            policies = (('baseline', False, False, False), ('rotary', False, False, True),
                        ('residual_rotary', True, False, True), ('nonlinear_rotary', False, True, True),
                        ('mixed_all', True, True, True))
        for name, residual, nonlinear, rotary in policies:
            executor.residual_dtype = torch.float32 if residual else torch.float16
            executor.precise = nonlinear; executor.rotary_precise = rotary
            executor.cos, executor.sin = (precise_cos, precise_sin) if rotary else (original_cos, original_sin)
            cases = []
            for ids, (forced, targets) in zip(prompts, traces):
                actual = executor.prefill(ids); errors = []; mismatches = []
                for step, token in enumerate(forced):
                    target = targets[step].cuda(); error = actual.float()-target
                    errors.append(dict(max_absolute=error.abs().max().item(), relative_rms=(error.square().mean()/target.square().mean()).sqrt().item()))
                    if actual.argmax(-1).item() != target.argmax(-1).item(): mismatches.append(step)
                    if step < 7: actual, _ = executor.decode(token)
                cases.append(dict(prompt_tokens=len(ids), errors=errors, greedy_mismatches=mismatches))
            elapsed = []
            for _ in range(3):
                torch.cuda.synchronize(); start = time.perf_counter(); executor.prefill(prompts[0])
                for token in traces[0][0][:7]: executor.decode(token)
                torch.cuda.synchronize(); elapsed.append(time.perf_counter()-start)
            result = dict(policy=name, cases=cases, eager_seconds=elapsed, eager_median=statistics.median(elapsed),
                          passed=all(not c['greedy_mismatches'] and all(e['max_absolute'] < .5 and e['relative_rms'] < .01 for e in c['errors']) for c in cases))
            results.append(result); print(json.dumps({'policy':name,'passed':result['passed'],'eager_seconds':result['eager_median']}), flush=True)
    Path(args.output).write_text(json.dumps(dict(device=torch.cuda.get_device_name(), gguf=args.gguf,
        reference='fixed-token HF FP32 eager over independently dequantized FP16 weights', results=results), indent=2)+'\n')


if __name__ == '__main__': main()

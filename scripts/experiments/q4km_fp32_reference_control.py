"""#583 scale-scan FP32-reference control for 8B-class dense Q4_K_M GGUF cells.

The ladder reference instrument (`scripts/verify_gguf_reference.py`) executes
its HF eager reference on the GPU; at 8B scale an FP32 reference (32 GB) fits no
GPU in this environment, so the recorded cells used `--reference-dtype float16`.
That FP16 reference arithmetic is itself a noise source at this gate: the
qwen3-8B cell measured 1.903% worst relative RMS against FP16 but 0.573% against
this FP32 control (the runtime passes; the FP16 reference added >1.3 points of
apparent RMS). Any 8B-class ladder cell therefore needs this control — or an
equivalent explicit FP32 bound — recorded alongside it.

Method: identical fixture to the ladder initial suite (two chat prompts, 129-
and 505-token raw prompts, eight decode steps), identical fp16 weight rounding
(the runner's `reference_weights(..., 'float16')`), but the reference executes
in FP32 on CPU. The runtime executes on the GPU under `guard_shared_cuda`. The
decode path is teacher-pinned to the reference's own greedy tokens so logits
compare step-aligned; graph replay was separately shown bit-exact against eager
in the paired ladder run, so a single eager pass is recorded here.

Usage (guarded, from the repository root on the labnode3 A30):

  PYTHONPATH=. python -m scripts.experiments.guard_shared_cuda \
      --gpu <A30-UUID> --timeout 1800 --report /tmp/guard.json -- \
      python scripts/experiments/q4km_fp32_reference_control.py \
      /path/to/checkpoint-Q4_K_M.gguf /path/to/hf-config-dir --output /tmp/control.json

Expect roughly 10-13 minutes of CPU reference per 8B checkpoint plus a few
minutes of GPU execution; host RSS peaks near 2x the checkpoint's dequantized
size (the state dict is built before the model materializes).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verify_gguf_reference import reference_weights  # noqa: E402
from damendaemon.gguf_model import GgufWeights  # noqa: E402
from damendaemon.runtime import make_executor  # noqa: E402
from damendaemon.device import device_name  # noqa: E402

STEPS = 8


def main():
    p = argparse.ArgumentParser()
    p.add_argument('gguf')
    p.add_argument('hf_config')
    p.add_argument('--output', required=True)
    p.add_argument('--context', type=int, default=512)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)

    source = GgufWeights(args.gguf, args.context, 'none')
    prompts = [source.chat_tokens([{'role': 'user', 'content': 'What is 2 + 2? Answer with one number.'}]),
               source.chat_tokens([{'role': 'user', 'content': 'Reply in French: Hello, how are you?'}])]
    base = source.raw_tokens('The ocean influences coastal weather. Explain the process in detail. ')
    for length in (129, args.context - 7):
        prompts.append((base * ((length + len(base) - 1) // len(base)))[:length])

    config = AutoConfig.from_pretrained(args.hf_config, local_files_only=True, trust_remote_code=False)
    config.tie_word_embeddings = False
    with no_init_weights():
        reference = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='eager')
    state, formats, embedding_type = reference_weights(args.gguf, 'float16')
    reference.load_state_dict(state, strict=True, assign=True)
    del state
    reference.eval().to(dtype=torch.float32)  # CPU FP32 arithmetic on fp16-rounded weights

    traces = []
    out = None
    with torch.inference_mode():
        for ids in prompts:
            out = reference(torch.tensor([ids]))
            trace = []
            for step in range(STEPS):
                trace.append(out.logits[:, -1, :].float())
                token = trace[-1].argmax(-1).item()
                if step < STEPS - 1:
                    out = reference(torch.tensor([[token]]), past_key_values=out.past_key_values, use_cache=True)
            traces.append((trace, [t.argmax(-1).item() for t in trace]))
        out = None
        del reference
    gc.collect()

    report = {'gguf': args.gguf, 'device': device_name(), 'torch': torch.__version__,
              'reference': 'CPU FP32 arithmetic, fp16-rounded weights (runner rounding), teacher-pinned to reference greedy',
              'graph': False, 'steps': STEPS, 'cases': []}
    with torch.inference_mode():
        executor = make_executor(source, use_graph=False)
        for index, ids in enumerate(prompts):
            trace, ref_tokens = traces[index]
            actual = executor.prefill(ids)[-1:]
            errors = []
            actual_tokens = []
            for step in range(STEPS):
                error = actual.float().cpu() - trace[step]
                rms = (error.square().mean().sqrt() / trace[step].square().mean().sqrt().clamp_min(1e-12)).item()
                errors.append(dict(step=step, relative_rms=rms, max_absolute=error.abs().max().item()))
                actual_tokens.append(int(actual.argmax(-1).item()))
                if step < STEPS - 1:
                    actual, _ = executor.decode(ref_tokens[step])
            report['cases'].append(dict(case=index, prompt_tokens=len(ids), errors=errors,
                                        reference_tokens=ref_tokens, actual_tokens=actual_tokens))
    report['worst_relative_rms'] = max(e['relative_rms'] for c in report['cases'] for e in c['errors'])
    report['greedy_agree_all'] = all(c['reference_tokens'] == c['actual_tokens'] for c in report['cases'])
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'worst_relative_rms': report['worst_relative_rms'], 'greedy_agree_all': report['greedy_agree_all']}))


main()

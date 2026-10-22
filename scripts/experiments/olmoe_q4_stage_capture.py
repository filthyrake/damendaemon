"""Layer-0 stage-local origin capture for the OLMoE Q4 near-tie diagnosis (#577).

Replays the 496-token prompt from the retained golden trace through the native
Q4 executor (guarded CUDA) and the HF reference (Q4_0 storage policy, FP32
arithmetic, CPU, eager attention), recording the hidden state at every stage
boundary of decoder layer 0:

  embedding | qkv input (norm1 out) | qkv output | attention core out
  (o_proj in) | o_proj out | mlp input (norm2 out) | mlp combined out

The first stage whose native-vs-reference relative RMS exceeds 1e-7 localizes
the sub-ppm divergence origin that later crosses the top-9 router boundary
(layer 0, row 239, gap 1.3e-8). The reference comparison for each stage is
row-aligned; packed prefill returns one logits row per prompt, so the whole
model comparison uses the prompt's last position only.

Usage:
  python -m scripts.experiments.olmoe_q4_stage_capture <trace.pt> <out.json>

Run under scripts/experiments/guard_shared_cuda.py on the qualified GPU with
DAMENDAEMON_QUANTIZATION=q4_0 and PYTHONPATH pointing at the tree root.
"""
import json
from pathlib import Path
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DAMENDAEMON_QUANTIZATION', 'q4_0')


@torch.inference_mode()
def main():
    trace_path, output = sys.argv[1], sys.argv[2]
    from scripts.experiments.moe_q4_model import install_validation_dispatch
    install_validation_dispatch()
    from reference_model import load_reference
    from damendaemon.qwen2 import ModelWeights
    from damendaemon.runtime import make_executor
    from damendaemon.device import synchronize

    trace = torch.load(trace_path, map_location='cpu', weights_only=True)
    event = next(e for e in trace['trace'] if e['kind'] == 'prefill'
                 and any(len(p) == 496 for p in e['prompts']))
    row = next(i for i, p in enumerate(event['prompts']) if len(p) == 496)
    ids = torch.tensor([event['prompts'][row]])
    print('prompt row', row, 'len', ids.shape[1], flush=True)

    source = ModelWeights(trace['path'], trace['context'])
    executor = make_executor(source, use_graph=False, max_batch=4)
    reference = load_reference(trace['path'], reference_quantization='q4_0', local_files_only=True,
                               trust_remote_code=False, torch_dtype=torch.float32,
                               attn_implementation='eager').to('cpu').eval()
    synchronize()

    # ---- reference captures (HF, CPU) ----
    ref = {}
    layer0 = reference.model.layers[0]

    def ref_hook(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.dim() == 3:
                value = value.squeeze(0)
            ref[name] = value.detach().clone()
        return hook

    def ref_pre_hook(name):
        def hook(module, inputs):
            value = inputs[0]
            if value.dim() == 3:
                value = value.squeeze(0)
            ref[name] = value.detach().clone()
        return hook

    reference.model.embed_tokens.register_forward_hook(ref_hook('embedding'))
    layer0.input_layernorm.register_forward_hook(ref_hook('qkv_in'))
    layer0.self_attn.o_proj.register_forward_pre_hook(ref_pre_hook('attn_core'))
    layer0.self_attn.o_proj.register_forward_hook(ref_hook('o_proj_out'))
    layer0.mlp.register_forward_pre_hook(ref_pre_hook('mlp_in'))
    layer0.mlp.register_forward_hook(ref_hook('mlp_down'))

    ref_logits = reference(ids, use_cache=False).logits

    # ---- native captures (OlmoeExecutor, CUDA) ----
    nat = {}
    wid = {name: id(executor.w[name]) for name in ['0.qkv.weight', '0.out', '0.router']}

    original_linear = type(executor)._linear
    original_prefill_attention = type(executor).prefill_attention
    original_experts = type(executor).experts

    def linear_capture(self, x, weight, bias=None):
        value = original_linear(self, x, weight, bias)
        if x.shape[0] == 496:
            if id(weight) == wid['0.qkv.weight']:
                nat['qkv_in'] = x.detach().float().cpu()
                nat['qkv_out'] = value.detach().float().cpu()
            elif id(weight) == wid['0.out']:
                nat['attn_core'] = x.detach().float().cpu()
                nat['o_proj_out'] = value.detach().float().cpu()
            elif id(weight) == wid['0.router']:
                nat['mlp_in'] = x.detach().float().cpu()
        return value

    def experts_capture(self, hidden, layer, prefill):
        result = original_experts(self, hidden, layer, prefill)
        if layer == 0 and hidden.shape[0] == 496:
            nat['mlp_down'] = result.detach().float().cpu()
        return result

    def attn_capture(self, query, key, value, query_start=0):
        out = original_prefill_attention(self, query, key, value, query_start)
        if query.shape[-2] == 496 and query_start == 0 and 'sdpa_out' not in nat:
            nat['sdpa_out'] = out.detach().float().cpu()
        return out

    type(executor)._linear = linear_capture
    type(executor).experts = experts_capture
    type(executor).prefill_attention = attn_capture
    try:
        native_logits = type(executor).prefill_batch(executor, event['prompts'], event['slots'])
    finally:
        type(executor)._linear = original_linear
        type(executor).experts = original_experts
        type(executor).prefill_attention = original_prefill_attention
    synchronize()

    # embedding is a pure lookup: recompute it exactly as _forward does.
    nat['embedding'] = F.embedding(ids.view(-1).to(executor.device),
                                   executor.w['embed']).float().cpu()

    def metrics(native, ref_tensor):
        native = native.detach().float().cpu()
        ref_tensor = ref_tensor.detach().float().cpu()
        if native.shape != ref_tensor.shape:
            return dict(shape_native=list(native.shape), shape_ref=list(ref_tensor.shape))
        error = native - ref_tensor
        rms = (error.square().mean().sqrt()
               / ref_tensor.square().mean().sqrt().clamp_min(1e-12)).item()
        return dict(relative_rms=rms, max_absolute=error.abs().max().item())

    report = {'model': trace['path'], 'prompt_row': row, 'prompt_len': int(ids.shape[1]),
              'stages': []}
    for stage in ('embedding', 'qkv_in', 'qkv_out', 'attn_core', 'o_proj_out',
                  'mlp_in', 'mlp_down'):
        if stage in nat and stage in ref:
            report['stages'].append(dict(stage=stage, **metrics(nat[stage], ref[stage])))
    if 'sdpa_out' in nat and 'attn_core' in nat:
        # internal consistency: the native o_proj input must be exactly the
        # reshaped SDPA output of the same (first, layer-0) attention call
        sdpa = nat['sdpa_out'].transpose(1, 2).reshape(nat['sdpa_out'].shape[2], -1)
        report['stages'].append(dict(stage='sdpa_vs_o_proj_in_native',
                                     **metrics(sdpa, nat['attn_core'])))
    # packed prefill returns one logits row per prompt at its last position
    report['logits'] = metrics(native_logits[row], ref_logits[0][-1])
    report['completed'] = True
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


main()

"""#643 probe: why did the base-arm forward device-assert at 32K?

Two constructions, one 32k-token forward each:
  A) device_map='cuda' + dtype=  (the crashed cell arm)
  B) load_gate_model's construction: torch_dtype + attn sdpa + .to('cuda')
Prints the resolved class and the outcome; asserts nothing except shape.
"""
import json
import os
import sys

if not __debug__:
    raise RuntimeError('probe asserts; run without -O')

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate import apply_chat_template, split_state_span
from tokengate.datasets.tasks import get_task
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE = "Qwen/Qwen3.5-2B"
rows = [json.loads(l) for l in open(os.path.join(
    _REPO_ROOT, "datasets", "niah_kamradt", "niah_kamradt_32k.jsonl"))][:1]
tok = AutoTokenizer.from_pretrained(BASE)
task = get_task("niah_kamradt")
ids = apply_chat_template(
    tok, task.format_prompt("{state}\n\nQuestion: " + rows[0]["question"]),
    state=rows[0]["body"])
print(json.dumps(dict(event="meta", n_tokens=len(ids))), flush=True)


def try_forward(label, **kw):
    print(f"[probe] construction {label}: {kw}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    model.eval()
    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
    print(f"[probe] class={type(model).__name__} "
          f"n_params={sum(p.numel() for p in model.parameters())}", flush=True)
    ids_t = torch.tensor(ids, device="cuda").unsqueeze(0)
    with torch.no_grad():
        out = model(ids_t)
    print(json.dumps(dict(event=label, ok=True,
                          logits_shape=list(out.logits.shape),
                          class_name=type(model).__name__)), flush=True)
    del model
    torch.cuda.empty_cache()


label = sys.argv[1] if len(sys.argv) > 1 else "A_device_map"
target_len = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if target_len and len(ids) > target_len:
    ids = ids[:target_len]
print(json.dumps(dict(event="meta", n_tokens=len(ids))), flush=True)
constructions = {"A_device_map": dict(dtype=torch.bfloat16, device_map="cuda"),
                 "B_gate_loader_style": dict(torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa")}
try:
    try_forward(label, **constructions[label])
except Exception as exc:
    print(json.dumps(dict(event=label, ok=False,
                          error=type(exc).__name__, detail=str(exc)[:160])),
          flush=True)

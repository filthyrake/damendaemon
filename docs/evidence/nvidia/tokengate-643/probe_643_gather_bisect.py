"""#643 probe 2: is the device assert the embedding gather (bad ids) or deeper?

CPU-audit the tokenized ids, then run a bare embedding gather on CUDA,
then one decoder layer — bisecting where the unpatched path asserts.
"""
import json
import os
import sys

if not __debug__:
    raise RuntimeError('probe asserts; run without -O')

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tokengate import apply_chat_template
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
    state=rows[0]["body"])[:4096]
print(json.dumps(dict(event="ids", n=len(ids), max_id=max(ids), min_id=min(ids),
                      vocab=tok.vocab_size)), flush=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
emb = model.model.embed_tokens
ids_t = torch.tensor(ids, device="cuda")
emb_out = emb(ids_t)
torch.cuda.synchronize()
print(json.dumps(dict(event="embedding_gather", ok=True,
                      shape=list(emb_out.shape),
                      vocab_dim=emb.num_embeddings)), flush=True)

hidden = emb_out.unsqueeze(0)
with torch.no_grad():
    layer0 = model.model.layers[0]
    out = layer0(hidden, attention_mask=None)
torch.cuda.synchronize()
print(json.dumps(dict(event="layer0", ok=True, shape=list(out[0].shape))), flush=True)

#!/usr/bin/env python3
"""#643 payoff cell: TokenGate Qwen3.5-2B (hybrid) NIAH 32K — the paper's
headline length-generalization setting, evaluated on a single RTX 3090.

First honest evaluation of the released hybrid checkpoint at 32K: #640's
hybrid arm was blocked by three upstream bugs (now patched in this clone,
cumulative #640+#642 patch set); this cell runs the same patched harness
path (generate_batch, prune_below_gate=True) the dense #640 numbers used.

Arms (16 deterministic rows of the 32k-tier pool, first-16 convention;
each body is roughly 32k ids long, the checkpoints were trained on the
16k tier, and this is the 2x length-generalization setting that
upstream's own demo uses):
  base     Qwen/Qwen3.5-2B plain HF generate (ungated control; prefill-wall
           TTFT from a separate bare forward per row)
  no-prune gate-adapted checkpoint, full context (the fine-tuning delta)
  2000/500/100  gated prefill at each K budget

Metrics per arm: accuracy (task.evaluate), TTFT = prefill wall (generate_batch
res.prefill_s; base arm: bare forward wall), needle-words-kept fraction (gate
arms), peak CUDA memory (warmup row first — flex compile + CUDA init — then
reset_peak_memory_stats, then the measured rows).

Containment (#643): bounded rows, absolute per-row deadline (120 s), atomic
report (tmp+rename), per-row progress prints, single invocation under the
1800 s class cap via the launcher's `timeout`.
"""
import json
import os
import sys
import time

if not __debug__:
    raise RuntimeError('cell asserts; run without -O')

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

HUB_HYBRID = "liana313/tokengate-niah-qwen3.5-2b"
BASE_MODEL = "Qwen/Qwen3.5-2B"
POOL = os.path.join(_REPO_ROOT, "datasets", "niah_kamradt", "niah_kamradt_32k.jsonl")
N_ROWS = 16
K_VALUES = [2000, 500, 100]
MAX_NEW = 32
ROW_DEADLINE_S = 120.0
REPORT = os.environ.get("TG643_REPORT", os.path.join(_REPO_ROOT, "tg643-cell.json"))


def atomic_dump(obj):
    tmp = REPORT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, REPORT)


def main():
    import torch
    from tokengate import apply_chat_template, load_model, split_state_span
    from tokengate.inference import _NO_PRUNING
    from tokengate.e2e_gate.decode.gate_decode import generate_batch
    from tokengate.datasets.tasks import get_task
    from tokengate.datasets.tasks.longbench import _normalize
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report = {"event": "meta", "torch": torch.__version__,
              "device": torch.cuda.get_device_name(0),
              "hybrid_checkpoint": HUB_HYBRID, "base_model": BASE_MODEL,
              "n_rows": N_ROWS, "k_values": K_VALUES,
              "max_new_tokens": MAX_NEW, "row_deadline_s": ROW_DEADLINE_S,
              "arms": {}}

    task = get_task("niah_kamradt")
    rows = [json.loads(l) for l in open(POOL)][:N_ROWS]
    assert len(rows) == N_ROWS, len(rows)

    # ---------------- arm 1: base model, plain HF generate ----------------
    print("[cell] loading base model", flush=True)
    from tokengate.e2e_gate.attn.qwen3_flex_attention import \
        patch_model_with_flex_attention
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    # Mandatory prep for any standalone use of the Qwen3.5 hybrid in this
    # tree (the qualified path applies it inside load_gate_model:711).
    base = patch_model_with_flex_attention(base)
    base.eval()

    def base_prompt_ids(s):
        return apply_chat_template(
            tok, task.format_prompt("{state}\n\nQuestion: " + s["question"]),
            state=s["body"])

    raw_ids = [base_prompt_ids(s) for s in rows]
    # apply_chat_template emits sentinel ids around the state span (negative
    # and >vocab); split_state_span strips them and returns (clean, start,
    # end). ALL arms — base included — must consume the cleaned ids, both
    # for validity and for identical prompts across arms.
    clean = [split_state_span(seq) for seq in raw_ids]

    arm = {"n": 0, "acc": [], "ttft": [], "timeouts": 0}
    # warmup: one bare forward (flex compile + CUDA init), unmeasured, then
    # fresh peak tracking — the same warmup/reset pattern as the gate arms.
    with torch.no_grad():
        base(torch.tensor(clean[0][0], device="cuda").unsqueeze(0))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i, (s, (cids, st, en)) in enumerate(zip(rows, clean)):
        deadline = time.monotonic() + ROW_DEADLINE_S
        ids_t = torch.tensor(cids, device="cuda").unsqueeze(0)
        # prefill wall: one bare forward (no generation) on the full prompt.
        # logits_to_keep=1 matches the gate arms' prefill (last-position
        # logits only) — without it the forward materializes 32k x 248k
        # full-vocab logits (~15.8 GiB, ~0.3 s), polluting both TTFT and
        # the peak-memory comparison.
        t0 = time.perf_counter()
        with torch.no_grad():
            base(ids_t, logits_to_keep=1)
        ttft = time.perf_counter() - t0
        # accuracy: plain greedy generate on the same prompt
        with torch.no_grad():
            out = base.generate(ids_t, max_new_tokens=MAX_NEW, do_sample=False)
        ans = tok.decode(out[0, len(cids):], skip_special_tokens=True)
        arm["acc"].append(float(task.evaluate(ans, s["answer"])))
        arm["ttft"].append(ttft)
        arm["n"] += 1
        if time.monotonic() > deadline:
            arm["timeouts"] += 1
        print(f"[base] row {i+1}/{N_ROWS} acc={arm['acc'][-1]:.0f} "
              f"ttft={ttft:.2f}s", flush=True)
    arm["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    arm["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    report["arms"]["base"] = arm
    atomic_dump(report)
    del base
    torch.cuda.empty_cache()

    # ---------------- gate arms: patched harness path ----------------
    print("[cell] loading hybrid gate checkpoint", flush=True)
    model = load_model(HUB_HYBRID, gate_attn="flex")
    gtok = AutoTokenizer.from_pretrained(model.base_model)

    def gate_run(label, k):
        arm = {"n": 0, "acc": [], "ttft": [], "needle_kept": [], "timeouts": 0}
        # warmup row (flex compile + CUDA init), not measured
        cids, st, en = clean[0]
        generate_batch(model.model, model.gate, gtok, [cids], [st], [en],
                       model.gate_layer, _NO_PRUNING if k is None else k,
                       MAX_NEW, prune_below_gate=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for i, (s, (cids, st, en)) in enumerate(zip(rows, clean)):
            deadline = time.monotonic() + ROW_DEADLINE_S
            t_row = time.perf_counter()
            res = generate_batch(
                model.model, model.gate, gtok, [cids], [st], [en],
                model.gate_layer,
                _NO_PRUNING if k is None else k,
                MAX_NEW, prune_below_gate=True)
            ans = res.answers[0]
            arm["acc"].append(float(task.evaluate(ans, s["answer"])))
            arm["ttft"].append(res.prefill_s)
            kept_state = [cids[p] for p in res.kept_pos[0][:res.n_kepts[0]]
                          if st <= p < en]
            kept_words = set(_normalize(gtok.decode(kept_state)).split())
            needle_words = _normalize(s["answer"]).split()
            arm["needle_kept"].append(
                sum(w in kept_words for w in needle_words) / len(needle_words))
            arm["n"] += 1
            if time.perf_counter() - t_row > ROW_DEADLINE_S:
                arm["timeouts"] += 1
            print(f"[{label}] row {i+1}/{N_ROWS} acc={arm['acc'][-1]:.0f} "
                  f"ttft={res.prefill_s:.2f}s "
                  f"kept={arm['needle_kept'][-1]:.0%}", flush=True)
        arm["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
        arm["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["arms"][label] = arm
        atomic_dump(report)

    if os.environ.get("TG643_BASE_ONLY") != "1":
        gate_run("no-prune", None)
        for k in K_VALUES:
            gate_run(str(k), k)

    # ---------------- summary ----------------
    print("\n[cell] summary (acc / mean TTFT s / peak alloc GiB)", flush=True)
    base_ttft = sum(report["arms"]["base"]["ttft"]) / N_ROWS
    for label, arm in report["arms"].items():
        acc = sum(arm["acc"]) / arm["n"]
        ttft = sum(arm["ttft"]) / len(arm["ttft"])
        speedup = base_ttft / ttft
        mem_ratio = (report["arms"]["base"]["peak_alloc_gib"]
                     / arm["peak_alloc_gib"])
        print(f"  {label:>8}: acc={acc:.3f} ttft={ttft:.3f}s "
              f"({speedup:.2f}x) peak={arm['peak_alloc_gib']:.2f} GiB "
              f"({mem_ratio:.2f}x)", flush=True)
    report["summary"] = {
        "base_mean_ttft_s": base_ttft,
        "arms": {label: {
            "acc": sum(a["acc"]) / a["n"],
            "ttft_mean_s": sum(a["ttft"]) / len(a["ttft"]),
            "peak_alloc_gib": a["peak_alloc_gib"],
            "timeouts": a["timeouts"],
            "needle_kept_mean": (sum(a["needle_kept"]) / len(a["needle_kept"])
                                 if a.get("needle_kept") else None),
        } for label, a in report["arms"].items()}}
    atomic_dump(report)
    print(f"[cell] wrote {REPORT}", flush=True)
    assert not any(a["timeouts"] for a in report["arms"].values()), "row deadline exceeded"


if __name__ == "__main__":
    main()

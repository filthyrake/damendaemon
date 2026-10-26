#!/usr/bin/env python3
"""#640 hybrid runner: TokenGate Qwen3.5-2B (hybrid, gate at DeltaNet layer 1)
NIAH-32K K-sweep, calling generate_batch directly with prune_below_gate=True
(the upstream eval harness's own flag — the public demo API omits it and
crashes on the hybrid because layers <= gate hold no KV entries).

Mirrors examples/niah_demo.py exactly otherwise (same samples, prompt
assembly, accuracy/needle-survival computation).
"""
import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

HUB_REPO = "liana313/tokengate-niah-qwen3.5-2b"
DEFAULT_SAMPLES = os.path.join(_REPO_ROOT, "examples", "data",
                               "niah_32k_samples.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=HUB_REPO)
    ap.add_argument("--samples", default=DEFAULT_SAMPLES)
    ap.add_argument("--k", type=int, nargs="+", default=[2000, 500, 100])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--gate-attn", default="flex", choices=["fa2", "flash", "flex"])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    from tokengate import apply_chat_template, load_model, split_state_span
    from tokengate.inference import _NO_PRUNING
    from tokengate.e2e_gate.decode.gate_decode import generate_batch
    from tokengate.datasets.tasks import get_task
    from tokengate.datasets.tasks.longbench import _normalize
    from transformers import AutoTokenizer

    model = load_model(args.checkpoint, gate_attn=args.gate_attn)
    tok = AutoTokenizer.from_pretrained(model.base_model)

    task = get_task("niah_kamradt")
    samples = [json.loads(l) for l in open(args.samples)][: args.n]
    ids = [apply_chat_template(
               tok, task.format_prompt("{state}\n\nQuestion: " + s["question"]),
               state=s["body"])
           for s in samples]
    clean = [split_state_span(seq) for seq in ids]

    def _run(k):
        accs, ttfts, needle_kept = [], [], []
        for s, seq, (cids, st, en) in zip(samples, ids, clean):
            t0 = time.perf_counter()
            res = generate_batch(
                model.model, model.gate, tok, [cids], [st], [en],
                model.gate_layer,
                _NO_PRUNING if k is None else k,
                args.max_new_tokens,
                prune_below_gate=True)
            prefill_s = res.prefill_s
            ans = res.answers[0]
            accs.append(task.evaluate(ans, s["answer"]))
            ttfts.append(prefill_s)
            kept_state = [cids[p] for p in res.kept_pos[0][:res.n_kepts[0]]
                          if st <= p < en]
            kept_words = set(_normalize(tok.decode(kept_state)).split())
            needle_words = _normalize(s["answer"]).split()
            needle_kept.append(
                sum(w in kept_words for w in needle_words) / len(needle_words))
        return (sum(accs) / len(accs), sum(ttfts) / len(ttfts),
                sum(needle_kept) / len(needle_kept))

    depths = ", ".join(f"{s['depth_actual']:.0%}" for s in samples)
    print(f"\n[niah-hybrid] {len(samples)} haystacks of ~32k tokens "
          f"(depths {depths}), prune_below_gate=True")
    print(f"\n{'kept tokens':>12} {'acc':>6} {'TTFT(ms)':>10} {'speedup':>8} "
          f"{'needle kept':>12}")
    print("-" * 54)
    full_ttft = None
    for label, k in [("all", None)] + [(str(k), k) for k in args.k]:
        acc, ttft, needle_frac = _run(k)
        full_ttft = full_ttft or ttft
        print(f"{label:>12} {acc:>6.2f} {ttft * 1e3:>10.1f} "
              f"{full_ttft / ttft:>7.2f}x {needle_frac:>11.0%}")


if __name__ == "__main__":
    main()

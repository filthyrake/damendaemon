"""#536 allocator-gap characterization — CPU-only, no sustained GPU load.

Compares the planner's required_memory_bytes against observed nvidia-smi used
memory from the guarded 8K attempts. Everything here runs on CPU against the
already-committed plan JSONs, so it never touches the thermal envelope.

Usage:
  python -m scripts.experiments.hybrid_allocator_gap_probe \
    --plans docs/evidence/distributed/qwen38-pipeline-plan-536/plan-8k.json \
             docs/evidence/distributed/qwen38-context-8k-536/dd536-plan-8192.json \
    --out docs/evidence/distributed/qwen38-context-8k-536/allocator_gap_report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Observed bands are baked for reproducibility; provenance in comments.
# Override via --observed-json if re-baselining.
OBSERVED = {
    # 2048: measured gate-2 passing values (docs/evidence/distributed/qwen38-two-rank-536/README.md + dd536-two-rank.json: A30 13555 MiB, A2 4830 MiB)
    2048: {"stage0_used_mib": (13555, 13555), "stage1_used_mib": (4830, 4830)},
    # 8192: five thermal-gated attempts (docs/evidence/distributed/qwen38-context-8k-536/README.md table: A30 free 4340-4782 / total 24576, A2 63-64 C vs guard <65)
    8192: {"stage0_used_mib": (19400, 22300), "stage1_used_mib": (11100, 14200)},
}

KNOWN_OVERHEADS_COMMENTARY = """Unmodeled categories (all outside planner's required_memory):
- CUDA context + driver (~350-600 MiB per device, nvidia-smi used includes this).
- Caching allocator reserved-vs-allocated fragmentation: allocator retains freed
  blocks across the 64 chunk iterations; nvidia-smi used tracks reserved.
- FP32/FP16 transient casts and Triton autotune scratch per chunk iteration.
- NCCL / IPC handles and Triton compilation footprints (one-time ~hundreds MiB).
- Planner counts 2*batch*context*hidden_size*4 as activation workspace, but real
  attention kernels allocate per-chunk temp Buffers proportional to chunk*context
  and keep them resident until empty_cache().
"""


def summarize_plan(path: pathlib.Path):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    placement = data.get("placement") or data
    ctx = data.get("context", placement.get("context", "?"))
    stages = placement.get("stages")
    if not isinstance(stages, list) or not stages:
        raise SystemExit(f"{path}: missing placement.stages")
    rows = []
    for s in stages:
        for key in ("layer_start", "layer_end", "weight_bytes", "scale_bytes", "state_bytes",
                     "activation_workspace_bytes", "output_head_workspace_bytes",
                     "workspace_floor_bytes", "required_memory_bytes"):
            if key not in s:
                raise SystemExit(f"{path}: stage missing '{key}'")
        rows.append({
            "layers": f"{s['layer_start']}-{s['layer_end']}",
            "weight_bytes": s["weight_bytes"],
            "scale_bytes": s["scale_bytes"],
            "state_bytes": s["state_bytes"],
            "activation_workspace_bytes": s["activation_workspace_bytes"],
            "output_head_workspace_bytes": s["output_head_workspace_bytes"],
            "workspace_floor_bytes": s["workspace_floor_bytes"],
            "required_memory_bytes": s["required_memory_bytes"],
            "required_mib": s["required_memory_bytes"] / (1024**2),
        })
    return {"path": str(path), "context": ctx, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", required=True, help="plan-*.json files")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--observed-json", type=pathlib.Path, default=None,
                    help="optional JSON overriding OBSERVED (context->stage->(lo,hi))")
    args = ap.parse_args()

    observed = OBSERVED
    if args.observed_json:
        observed = {int(k): v for k, v in json.loads(args.observed_json.read_text()).items()}

    report = {"generated_by": "hybrid_allocator_gap_probe.py", "cpu_only": True, "plans": [], "gap": []}
    seen_gap_keys: set[tuple] = set()

    for p in args.plans:
        summ = summarize_plan(pathlib.Path(p))
        report["plans"].append(summ)
        ctx = summ["context"]
        obs = observed.get(ctx) if isinstance(ctx, int) else None
        print(f"\n=== {p}  context={ctx} ===")
        for i, r in enumerate(summ["rows"]):
            print(f" stage {i} layers {r['layers']}: "
                  f"req {r['required_mib']:.0f} MiB "
                  f"(w {r['weight_bytes']/2**20:.0f} + scale {r['scale_bytes']/2**20:.0f} + "
                  f"state {r['state_bytes']/2**20:.0f} + act {r['activation_workspace_bytes']/2**20:.0f} + "
                  f"head {r['output_head_workspace_bytes']/2**20:.0f} + floor {r['workspace_floor_bytes']/2**20:.0f})")
        if obs:
            for i, r in enumerate(summ["rows"]):
                key = f"stage{i}_used_mib"
                if key not in obs:
                    continue
                gap_key = (ctx, i)
                if gap_key in seen_gap_keys:
                    continue  # dedup same context appearing in multiple plan files
                seen_gap_keys.add(gap_key)
                lo, hi = obs[key]
                mid = (lo + hi) / 2
                gap_mid = mid - r["required_mib"]
                pct = gap_mid / r["required_mib"] * 100 if r["required_mib"] else 0
                print(f"  observed used {lo:.0f}-{hi:.0f} MiB (mid {mid:.0f}); gap {gap_mid:.0f} MiB ({pct:.0f}% overhead)")
                report["gap"].append({
                    "context": ctx, "stage": i,
                    "planner_required_mib": round(r["required_mib"]),
                    "observed_used_mib_lo": lo, "observed_used_mib_hi": hi,
                    "gap_mid_mib": round(gap_mid), "overhead_pct_mid": round(pct),
                })
        elif isinstance(ctx, int):
            print("  (no observed band for this context)")

    report["notes"] = KNOWN_OVERHEADS_COMMENTARY
    report["recommendations"] = [
        "Expose torch.cuda.memory_summary per stage after a synthetic ramp on a cold run (outside thermal retry) to split reserved vs allocated.",
        "Insert torch.cuda.empty_cache() between chunk groups (e.g., every 16 chunks) and measure whether A2 peak drops toward plan + CUDA context.",
        "Retry thermal cell with inter-chunk cooldown or lower duty-cycle, not unconditional hot loop.",
    ]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.out)
        print(f"\nwrote {args.out}")

    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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
import sys


OBSERVED = {
    # 2048: measured gate-2 passing values (docs/evidence/distributed/qwen38-two-rank-536/)
    2048: {"stage0_used_mib": (13555, 13555), "stage1_used_mib": (4830, 4830)},
    # 8192: five thermal-gated attempts, A30 free 4340-4782 / A30 total 24576, A2 temps 63-64C
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
    data = json.loads(path.read_text())
    placement = data.get("placement") or data
    ctx = data.get("context", placement.get("context", "?"))
    stages = placement["stages"]
    rows = []
    for s in stages:
        rows.append({
            "layers": f"{s['layer_start']}–{s['layer_end']}",
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
    args = ap.parse_args()

    report = {"generated_by": "hybrid_allocator_gap_probe.py", "cpu_only": True, "plans": [], "gap": []}

    for p in args.plans:
        summ = summarize_plan(pathlib.Path(p))
        report["plans"].append(summ)
        ctx = summ["context"]
        obs = OBSERVED.get(ctx)
        # pretty print
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
                if key in obs:
                    lo, hi = obs[key]
                    mid = (lo+hi)/2
                    gap_mid = mid - r['required_mib']
                    pct = gap_mid / r['required_mib'] * 100
                    print(f"  observed used {lo:.0f}–{hi:.0f} MiB (mid {mid:.0f}); gap {gap_mid:.0f} MiB ({pct:.0f}% overhead)")
                    report["gap"].append({
                        "context": ctx, "stage": i,
                        "planner_required_mib": round(r['required_mib']),
                        "observed_used_mib_lo": lo, "observed_used_mib_hi": hi,
                        "gap_mid_mib": round(gap_mid), "overhead_pct_mid": round(pct),
                    })

    report["notes"] = KNOWN_OVERHEADS_COMMENTARY
    report["recommendations"] = [
        "Expose torch.cuda.memory_summary per stage after a synthetic ramp on a cold run (outside thermal retry) to split reserved vs allocated.",
        "Insert torch.cuda.empty_cache() between chunk groups (e.g., every 16 chunks) and measure whether A2 peak drops toward plan + CUDA context.",
        "Retry thermal cell with inter-chunk cooldown or lower duty-cycle, not unconditional hot loop.",
    ]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")

    # Always dump report preview to stdout
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Plan declared CPU reference storage from headers; never load weights or probe devices."""
import argparse
import json
import os
import tempfile
from pathlib import Path
from dllm.reference_stage_plan import inspect_reference_storage,plan_reference_stages
from scripts.serve_cpu_reference import bounded_integer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family',choices=('glm','flash','deepseek'));parser.add_argument('snapshot')
    parser.add_argument('--budget-mib',nargs='+',type=bounded_integer(1,2**20),required=True)
    parser.add_argument('--context',type=bounded_integer(1,2**20),default=512)
    parser.add_argument('--slots',type=bounded_integer(1,4),default=4)
    parser.add_argument('--prefill-chunk',type=bounded_integer(1,4096),default=128)
    parser.add_argument('--plan-output',help='write ranges-only JSON for the existing host command, only if a plan fits')
    for name,default in (('weight',256),('cache',256),('projection-workspace',8),('workspace',64),('output',64),('frame',64)):
        parser.add_argument('--'+name+'-mib',type=bounded_integer(1,64 if name=='frame' else 2048),default=default)
    args=parser.parse_args(argv)
    try:
        if not 2<=len(args.budget_mib)<=8:raise ValueError('storage planning requires 2..8 ordered CPU budgets')
        options=dict(projection_workspace_bytes=args.projection_workspace_mib*2**20,
            max_workspace_bytes=args.workspace_mib*2**20,max_output_bytes=args.output_mib*2**20)
        layout=inspect_reference_storage(args.snapshot,args.family,**options)
        report=plan_reference_stages(layout,[v*2**20 for v in args.budget_mib],context=args.context,slots=args.slots,
            chunk_size=args.prefill_chunk,max_weight_bytes=args.weight_mib*2**20,max_cache_bytes=args.cache_mib*2**20,
            frame_bytes=args.frame_mib*2**20,**options)
        if args.plan_output is not None and report['storage_plan_found']:
            destination=Path(args.plan_output).resolve()
            if str(destination) in {record[0] for record in layout.receipt}:
                raise ValueError('plan output cannot replace an inspected checkpoint file')
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(mode='w',dir=destination.parent,delete=False) as writer:
                    temporary=Path(writer.name);writer.write(json.dumps(report['plan'],indent=2)+'\n')
                layout.verify_unchanged();os.replace(temporary,destination)
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
        print(json.dumps(report,sort_keys=True),flush=True)
        return 0 if report['storage_plan_found'] else 1
    except (ValueError,OSError,KeyError,TypeError) as error:parser.exit(2,f'CPU reference storage planning failed: {error}\n')


if __name__=='__main__':raise SystemExit(main())

"""Audit declared CPU/CUDA range-plan storage from headers without loading weights."""
import argparse
import json
from damendaemon.cpu_reference_text import read_json
from damendaemon.reference_stage_plan import inspect_reference_storage
from damendaemon.cuda_host_storage import audit_cuda_host_storage
from scripts.serve_cpu_reference import bounded_integer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family',choices=('glm','flash','deepseek'));parser.add_argument('snapshot')
    parser.add_argument('--plan',required=True,help='existing ranges-only JSON; never rewritten')
    for name in ('host-budget-mib','device-budget-mib'):
        parser.add_argument('--'+name,nargs='+',required=True,type=bounded_integer(1,2**20))
    parser.add_argument('--context',type=bounded_integer(1,2**20),default=512)
    parser.add_argument('--slots',type=bounded_integer(1,4),default=4)
    parser.add_argument('--prefill-chunk',type=bounded_integer(1,4096),default=128)
    parser.add_argument('--input-kind',choices=('tokens','flash_embeddings'),default='tokens')
    for name,default in (('weight',256),('device-weight',256),('cache',256),('workspace',64),('projection-workspace',8),
                         ('device-workspace',256),('output',64),('transfer',64),('frame',64)):
        parser.add_argument('--'+name+'-mib',type=bounded_integer(1,64 if name=='frame' else 2048),default=default)
    args=parser.parse_args(argv)
    try:
        plan=read_json(args.plan,1024**2)
        if not isinstance(plan,dict) or set(plan)!={'ranges'}:raise ValueError('audit plan must contain exactly ranges')
        layout=inspect_reference_storage(args.snapshot,args.family,projection_workspace_bytes=args.projection_workspace_mib*2**20,
            max_workspace_bytes=args.workspace_mib*2**20,max_output_bytes=args.output_mib*2**20)
        report=audit_cuda_host_storage(layout,plan['ranges'],[v*2**20 for v in args.host_budget_mib],
            [v*2**20 for v in args.device_budget_mib],context=args.context,slots=args.slots,chunk_size=args.prefill_chunk,
            max_weight_bytes=args.weight_mib*2**20,max_device_bytes=args.device_weight_mib*2**20,max_cache_bytes=args.cache_mib*2**20,
            device_workspace_bytes=args.device_workspace_mib*2**20,max_output_bytes=args.output_mib*2**20,
            max_transfer_bytes=args.transfer_mib*2**20,frame_bytes=args.frame_mib*2**20,input_kind=args.input_kind)
        print(json.dumps(report,sort_keys=True),flush=True)
        return 0 if report['declared_limits_fit'] else 1
    except (ValueError,OSError,KeyError,TypeError) as error:parser.exit(2,f'CUDA declared storage audit failed: {error}\n')


if __name__=='__main__':raise SystemExit(main())

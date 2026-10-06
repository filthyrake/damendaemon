"""Replay an independent trace and prove both GPU stages used packed prefill."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--output',required=True)
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),required=True)
    p.add_argument('--no-graph',action='store_true')
    p.add_argument('--sequential',action='store_true',help='control: disable packing on the identical trace');args=p.parse_args()
    os.environ.update(DLLM_EXPERIMENTAL_LAYER_SPLIT='12',DLLM_PARTITION_PACKED_PREFILL=str(int(not args.sequential)),
                      DLLM_QUANTIZATION=args.quantization)
    from dllm.layer_stage import LayerStage
    original=LayerStage.run_prefill_packed;calls=[]
    def observed(stage,inputs,lengths,slots):
        result=original(stage,inputs,lengths,slots)
        calls.append(dict(layers=list(stage.layer_range),lengths=list(lengths),slots=list(slots)))
        return result
    LayerStage.run_prefill_packed=observed
    previous=sys.argv
    try:
        sys.argv=['scripts/replay_reference.py',args.trace,*(['--no-graph'] if args.no_graph else [])]
        captured=io.StringIO()
        replay_exit=0
        with redirect_stdout(captured):
            try:runpy.run_path(sys.argv[0],run_name='__main__')
            except SystemExit as error:replay_exit=error.code
        report=json.loads(captured.getvalue())
        assert report['weight_quantization']==args.quantization
        if args.sequential:assert not calls
        else:assert calls and {tuple(call['layers']) for call in calls}=={(0,12),(12,24)}
        report.update(packed_stage_calls=calls,sequential_control=args.sequential,
            trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),scope='Real two-GPU packed prefill reference replay; not a performance comparison',
            sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                     ('dllm/batch.py','dllm/layer_stage.py','dllm/partition.py','dllm/partition_policy.py')})
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']),packed_stage_calls=len(calls),
            failures=[check for check in report['checks'] if not check['passed']])),flush=True)
        if replay_exit or not report['passed']:raise SystemExit(1)
    finally:sys.argv=previous;LayerStage.run_prefill_packed=original


if __name__=='__main__':main()

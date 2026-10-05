"""Run independent validation under the isolated Q4 OLMoE implementation."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from scripts.experiments.moe_q4_model import install_validation_dispatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--grouped-prefill', action='store_true')
    parser.add_argument('--eager-prefill-attention', action='store_true',
                        help='isolated FP32 attention arithmetic control; not a performance path')
    parser.add_argument('--fp64-norm', action='store_true',
                        help='isolated higher-precision normalization control')
    parser.add_argument('script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sys.path.insert(0,str(Path(args.script).resolve().parent))
    report = install_validation_dispatch()
    if args.fp64_norm:
        from scripts.experiments.moe_q4_norm import install_validation_dispatch as install_norm
        report['normalization'] = install_norm()
    if args.eager_prefill_attention:
        from scripts.experiments.moe_q4_attention import install_validation_dispatch as install_attention
        report['prefill_attention'] = install_attention()
    if args.grouped_prefill:
        from scripts.experiments.moe_q4_grouped import install_validation_dispatch as install_grouped
        report['grouped_prefill'] = install_grouped()
    report.update(script=args.script, arguments=args.arguments, completed=False)
    try:
        sys.argv = [args.script,*args.arguments]
        runpy.run_path(args.script,run_name='__main__')
        if not (report['model_loads'] or report['reference_loads']):
            raise AssertionError('Q4 OLMoE experiment was not exercised')
        report['completed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()

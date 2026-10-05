"""Run an existing validation script with isolated GPU grouped MoE prefill."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from scripts.experiments.moe_grouped import install_validation_dispatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    report = install_validation_dispatch()
    report.update(script=args.script, arguments=args.arguments, completed=False)
    try:
        sys.argv = [args.script, *args.arguments]
        runpy.run_path(args.script, run_name='__main__')
        if not report['selected_calls']:
            raise AssertionError('grouped MoE experiment was not selected')
        report['completed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':main()

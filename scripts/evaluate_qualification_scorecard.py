"""Evaluate local evidence attestations without running hardware or fetching URLs."""
import argparse
import json
from pathlib import Path
from dllm.checkpoint import _unique_fields
from dllm.qualification_scorecard import evaluate_scorecard,verify_evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--evidence-root',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if not args.input.is_file():raise ValueError('scorecard input must be a regular file')
        with args.input.open('rb') as reader:raw=reader.read(16*1024**2+1)
        if len(raw)>16*1024**2:raise ValueError('scorecard input exceeds 16 MiB')
        def invalid_constant(value):raise ValueError('nonfinite JSON constant: '+value)
        try:source=json.loads(raw,object_pairs_hook=_unique_fields,parse_constant=invalid_constant)
        except RecursionError as error:raise ValueError('scorecard JSON nesting is too deep') from error
        result=evaluate_scorecard(source)
        result['verified_evidence_files']=verify_evidence(source,args.evidence_root)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError) as error:parser.exit(2,f'Scorecard rejected: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())

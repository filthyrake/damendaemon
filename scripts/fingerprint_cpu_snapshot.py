"""Print a deterministic bounded identity for a contained local CPU snapshot."""
import argparse
import json
from dllm.cpu_snapshot_identity import fingerprint


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--max-mib',type=int,default=2048)
    parser.add_argument('--timeout',type=float,default=60)
    args=parser.parse_args(argv)
    try:print(json.dumps(fingerprint(args.snapshot,max_bytes=args.max_mib*1024**2,timeout=args.timeout).report(),indent=2))
    except (ValueError,OSError,RuntimeError) as error:parser.exit(2,f'CPU snapshot identity failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())

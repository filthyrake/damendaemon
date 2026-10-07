"""Audit pinned-layout GLM metadata without reading checkpoint tensor values."""
import argparse
import json
from dllm.glm_checkpoint import audit_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',help='local config and checkpoint metadata directory')
    parser.add_argument('--names-only',action='store_true',help='use index only; do not open shards')
    args=parser.parse_args()
    try:report=audit_snapshot(args.snapshot,names_only=args.names_only)
    except (ValueError,OSError) as error:parser.exit(2,f'checkpoint audit failed: {error}\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':main()

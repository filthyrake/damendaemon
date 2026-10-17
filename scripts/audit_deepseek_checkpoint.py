"""Audit pinned-layout DeepSeek metadata without materializing checkpoint weights."""
import argparse
import json

from damendaemon.deepseek_checkpoint import audit_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', help='local directory containing config and checkpoint metadata')
    parser.add_argument('--names-only', action='store_true', help='use tensor index only; do not open shards')
    args = parser.parse_args()
    try:
        report = audit_snapshot(args.snapshot, names_only=args.names_only)
    except (ValueError, OSError) as error:
        parser.exit(2, f'checkpoint audit failed: {error}\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

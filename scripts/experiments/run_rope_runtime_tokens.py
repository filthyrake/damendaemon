"""Opt in to the rotary stride experiment for existing qualification harnesses."""
import argparse
import runpy
import sys

from scripts.experiments.rope_runtime_tokens import install_validation_dispatch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=('replay', 'http', 'warm'))
    args, remaining = parser.parse_known_args()
    install_validation_dispatch()
    script = {
        'replay': 'scripts/verify_partition_packed.py',
        'http': 'scripts/verify_partition_service.py',
        'warm': 'scripts/experiments/bench_partition_rows.py',
    }[args.target]
    sys.argv = [script, *remaining]
    runpy.run_path(script, run_name='__main__')

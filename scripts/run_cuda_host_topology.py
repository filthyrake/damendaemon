"""Explicit NVIDIA CUDA native host topology worker/completion/HTTP command."""
import argparse
from scripts.run_cpu_host_topology import main as run_topology


def main(argv=None):
    selector=argparse.ArgumentParser(description=__doc__,add_help=False)
    selector.add_argument('--family',required=True,choices=('glm','flash','deepseek'))
    selected,remaining=selector.parse_known_args(argv)
    return run_topology(remaining,family=selected.family,cuda=True)


if __name__=='__main__':raise SystemExit(main())

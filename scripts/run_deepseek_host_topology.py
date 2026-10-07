"""Explicit native DeepSeek chat across caller-provisioned CPU host stages."""
from scripts.run_cpu_host_topology import main as _main


def main(argv=None):return _main(argv,family='deepseek')


if __name__=='__main__':raise SystemExit(main())

"""Explicit native Flash text/tool completion across bounded CPU host stages."""
from scripts.run_cpu_host_topology import main as _main


def main(argv=None):return _main(argv,family='flash')


if __name__=='__main__':raise SystemExit(main())

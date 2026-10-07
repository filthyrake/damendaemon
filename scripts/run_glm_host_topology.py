"""Explicit GLM raw completion across local or caller-provisioned CPU host stages."""
from scripts.run_cpu_host_topology import main as _main


def main(argv=None):
    return _main(argv,family='glm')


if __name__=='__main__':raise SystemExit(main())

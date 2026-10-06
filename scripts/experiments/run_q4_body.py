"""Explicit experimental dispatch for existing qualification and HTTP harnesses."""
import argparse
import runpy
import sys
from scripts.experiments.q4_body_single_dot import install_validation_dispatch


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('target',choices=('serve','replay','single-replay','rebind'))
    parser.add_argument('--body-kernel',choices=('single-dot','paired-load','adjacent-gate','adjacent-all','adjacent-inplace'),default='single-dot')
    args,remaining=parser.parse_known_args()
    install_validation_dispatch(args.body_kernel)
    if args.target=='serve':
        sys.argv=['dllm','serve',*remaining]
        runpy.run_module('dllm',run_name='__main__')
    else:
        script={'replay':'verify_partition_packed','single-replay':'replay_reference','rebind':'verify_partition_rebind'}[args.target]
        sys.argv=['scripts/'+script+'.py',*remaining]
        runpy.run_path(sys.argv[0],run_name='__main__')

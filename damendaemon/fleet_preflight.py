"""Read-only fleet host preflight for distributed rendezvous planning.

Bounded, host-local checks with one pass/fail per item and actionable text
encoding the observed fleet failure modes (gate 4): mixed torch/NCCL builds
failing rendezvous with opaque socket errors, ephemeral-port firewall gaps,
and blocked connects presenting as open-loop hangs. The local report cannot
see the peer's versions — compare two hosts' reports side by side before
rendezvous; the startup agreement (distributed_agreement) is the enforcement
point that rejects a mismatch explicitly.
"""
import json
import socket
import sys
import time


def torch_report(torch):
    """Build/NCCL/runtime item for the local torch installation."""
    hip = getattr(torch, 'version', None) and torch.version.hip
    build = 'rocm' if hip else ('cuda' if torch.version.cuda else 'cpu')
    report = dict(item='torch', passed=True, torch=torch.__version__, build=build,
                  cuda=torch.version.cuda, hip=hip)
    if build == 'cuda':
        nccl = tuple(torch.cuda.nccl.version())
        report['nccl'] = '.'.join(map(str, nccl))
        report['note'] = ('record this report and compare with the peer\'s before '
                          'rendezvous; mixed torch/NCCL builds fail with opaque '
                          'socket errors — align venvs first (gate 4 finding)')
    elif build == 'rocm':
        report['nccl'] = 'rccl (bundled)'
        report['note'] = 'RCCL versions are not exposed uniformly; rely on the startup agreement'
    else:
        report['nccl'] = None
    return report


def gpu_inventory(torch):
    """One item per visible GPU: name, UUID, total/free memory."""
    if not torch.cuda.is_available():
        return dict(item='gpus', passed=False, error='no CUDA/ROCm device visible; '
                    'check the driver and visibility variables (ROCR/HIP/CUDA_VISIBLE_DEVICES)')
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        uuid = getattr(properties, 'uuid', None)
        devices.append(dict(index=index, name=properties.name,
                            uuid=str(uuid) if uuid is not None else None,
                            total_gib=round(total/2**30, 2), free_gib=round(free/2**30, 2)))
    return dict(item='gpus', passed=bool(devices), devices=devices,
                note='two-rank distributed-run needs two distinct explicit GPU UUIDs')


def peer_reachability(host, port, timeout=5.0):
    """TCP connect to the peer's rendezvous port; classify the failure."""
    item = dict(item='peer', host=host, port=port)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            item['passed'] = True
            item['note'] = 'TCP reachable; NCCL also binds ephemeral source ports — '\
                           'keep the source-scoped firewall range open in both directions'
    except socket.timeout:
        item['passed'] = False
        item['note'] = ('connect timed out: a blocked connect presents as an open-loop '
                        'hang at rendezvous — open the port and the ephemeral range first')
    except OSError as error:
        item['passed'] = False
        item['note'] = f'connect refused/failed: {error.strerror or error}; is the peer listening?'
    return item


def run_report(torch, peer=None, probe_timeout=5.0):
    items = [torch_report(torch), gpu_inventory(torch)]
    if peer is not None:
        host, _, port = peer.partition(':')
        items.append(peer_reachability(host, int(port) if port else 29500, probe_timeout))
    return dict(items=items, passed=all(item['passed'] for item in items))


def run_report_with_cotenancy(torch, peer=None, probe_timeout=5.0):
    """Preflight plus the read-only co-tenancy (#609) and topology (#604) items."""
    report = run_report(torch, peer=peer, probe_timeout=probe_timeout)
    from .cotenancy_inventory import run_coTenancy_report
    cotenancy = run_coTenancy_report()
    report['items'].extend(cotenancy['items'])
    report['passed'] = report['passed'] and cotenancy['passed']
    try:
        from .topology_inventory import run_topology_report
        report['items'].append(run_topology_report(torch))
    except Exception as error:  # a broken probe must not sink the whole preflight
        report['items'].append(dict(item='topology', passed=False, error=str(error)))
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='damendaemon preflight', description=__doc__)
    parser.add_argument('--peer', metavar='HOST[:PORT]', default=None,
                        help='probe the peer\'s rendezvous TCP port (default port 29500)')
    parser.add_argument('--probe-timeout', type=float, default=5.0)
    parser.add_argument('--cotenancy', action='store_true', default=True,
                        help='include read-only NVIDIA co-tenancy items (MIG, compute mode, MPS)')
    args = parser.parse_args(argv)
    report = run_report_with_cotenancy(__import__('torch'), peer=args.peer,
                                       probe_timeout=args.probe_timeout)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

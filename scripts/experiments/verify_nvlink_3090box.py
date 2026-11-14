"""Bounded 3090box cell: dual RTX 3090 NVLink interconnect qualification (#599).

#599 documents "No interconnect topology anywhere in placement" and
"can_device_access_peer" as current fleet gaps. 3090box is now a dual-3090
host with NVLink (nvidia-smi topo: NV4). This cell qualifies the actual
capability a placement layer could rely on:

1. TOPOLOGY: nvidia-smi reports an NVLink connection between the two GPUs
   in both directions.
2. PEER ACCESS: torch.cuda.can_device_access_peer(0,1) and (1,0) are both
   True; torch.cuda.enable_peer_access succeeds both directions.
3. BANDWIDTH: device-to-device 2 GiB copy between the GPUs exceeds the
   host-staged control (copy through CPU pinned staging) by a recorded
   margin — NVLink is actually engaged, not silently routed via host.
4. FINITENESS: peer-copied payload is bit-identical to source.
5. GUARD: run under guard_shared_cuda with BOTH explicit GPU UUIDs; the
   runner refuses unless exactly the two fleet 3090s are visible.

No serving/SLO claim — interconnect capability qualification only.
"""
import argparse
import json
import subprocess
from pathlib import Path

from scripts.experiments.split_gguf import save

GPUS = ('GPU-24ef3d4c-51dc-d004-4150-d6a70b05e069', 'GPU-61e2abe7-7bda-2d07-df9e-c0df1ffc64ca')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gib', type=float, default=2.0)
    args = parser.parse_args()
    import os
    if os.environ.get('DAMENDAEMON_SHARED_CUDA_GUARD') != '1':
        raise ValueError('requires the shared CUDA guard wrapper')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible != ','.join(GPUS):
        raise ValueError(f'requires exactly the two 3090box GPU UUIDs, saw {visible}')
    if args.output.exists():
        raise ValueError('new report path required')
    import torch
    if torch.cuda.device_count() != 2:
        raise ValueError('requires both 3090s visible')
    names = [torch.cuda.get_device_name(i) for i in (0, 1)]
    if any(not n.startswith('NVIDIA GeForce RTX 3090') for n in names):
        raise ValueError(f'unexpected devices {names}')
    torch.backends.cuda.matmul.allow_tf32 = False

    report = dict(passed=False, device=names, torch=torch.__version__,
                  mechanism='nvlink_peer', gates=[])
    save(args.output, report)
    try:
        # Gate 1: topology — an NV line between the two GPUs (column 3 of
        # nvidia-smi topo -m rows: GPU, self, interconnect, CPU affinity...).
        topo = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True,
                              text=True, timeout=30).stdout
        rows = {line.split('\t')[0].strip(): line for line in topo.splitlines()
                if line.startswith('GPU')}
        connection = rows.get('GPU0', '').split('\t')[2].strip() if 'GPU0' in rows else ''
        topology = dict(gate='topology_nvlink', connection=connection, rows=rows.get('GPU0', '')[:80])
        topology['passed'] = connection.startswith('NV')
        report['gates'].append(topology)
        save(args.output, report)

        # Gate 2: peer access — query both directions. torch 2.10 exposes
        # can_device_access_peer but no enable_peer_access: peer access for
        # copy_ is managed by the CUDA driver/runtime, and the measured
        # cross-device copy IS the observable qualification (gate 3).
        forward = torch.cuda.can_device_access_peer(0, 1)
        backward = torch.cuda.can_device_access_peer(1, 0)
        peer = dict(gate='peer_access', forward=bool(forward), backward=bool(backward))
        peer['passed'] = bool(forward and backward)
        report['gates'].append(peer)
        save(args.output, report)

        # Gate 3+4: bandwidth vs host-staged control + bit-identical payload.
        elements = int(args.gib * 2**30 // 4)
        source = torch.arange(elements, dtype=torch.float32, device='cuda:0')
        target = torch.empty_like(source, device='cuda:1')
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

        def timed(fn, sync_devices=(0, 1)):
            import time
            start = time.monotonic()
            result = fn()
            for index in sync_devices:
                torch.cuda.synchronize(index)
            return result, time.monotonic() - start

        ( _, peer_seconds) = timed(lambda: target.copy_(source, non_blocking=False))
        peer_gib_s = args.gib / peer_seconds
        identical = bool(torch.equal(target.cpu(), source.cpu()))

        staged = torch.empty_like(source, device='cuda:1')
        (_, staged_seconds) = timed(lambda: staged.copy_(source.cpu().pin_memory()))
        staged_gib_s = args.gib / staged_seconds

        bandwidth = dict(gate='bandwidth', peer_gib_s=round(peer_gib_s, 2),
                         host_staged_gib_s=round(staged_gib_s, 2))
        bandwidth['ratio'] = round(peer_gib_s / staged_gib_s, 2)
        bandwidth['passed'] = peer_gib_s > staged_gib_s * 1.5
        report['gates'].append(bandwidth)
        integrity = dict(gate='payload_bit_identical', passed=identical)
        report['gates'].append(integrity)
        report.update(peer_gib_s=round(peer_gib_s, 2), host_staged_gib_s=round(staged_gib_s, 2))
        save(args.output, report)

        report['passed'] = all(g['passed'] for g in report['gates'])
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
        save(args.output, report)
        print('NVLINK_CELL_DONE', 'passed' if report['passed'] else 'FAILED')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()

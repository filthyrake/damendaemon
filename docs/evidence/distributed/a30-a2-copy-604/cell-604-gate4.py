"""#604 gate 4: measured A30->A2 boundary copy cell on the qualified pair.

Read-only on device state (no MIG, no persistence mode, nothing enabled
except the disclosed peer-access attempt the gate-3 policy itself makes).
Measurements are pure copy walls, reported separately from compute:
  - direct blocking-unpinned device-to-device copy (dst.copy_(src)), which
    the driver internally stages through pageable host memory when P2P is
    unavailable in this VM;
  - the pinned-staged path from gate 3: D2H into a pinned buffer, then
    non-blocking H2D onto the peer device (the boundary-contract transfer
    without the caller-side clone).
Medians over warm channels; both directions; 1/8/32 MiB payload sizes.
Plus the NUMA-local vs NUMA-remote host-staging characterization — or its
honest skip when the VM exposes a single NUMA domain.
"""
import json
import statistics
import time

import torch

from damendaemon.transfer_policy import peer_plan, enable_peer_access

if not __debug__:
    raise RuntimeError('gates on assertions; run without -O')

SIZES_BYTES = [1024 ** 2, 8 * 1024 ** 2, 32 * 1024 ** 2]
WARM = 5
MEASURED = 21
A30, A2 = 0, 1


def emit(**event):
    print(json.dumps(event), flush=True)


def sync_all(devices):
    for index in devices:
        torch.cuda.synchronize(index)


def median_seconds(fn):
    for _ in range(WARM):
        fn()
    sync_all((A30, A2))
    walls = []
    for _ in range(MEASURED):
        started = time.perf_counter()
        fn()
        sync_all((A30, A2))
        walls.append(time.perf_counter() - started)
    return statistics.median(walls)


@torch.inference_mode()
def copy_cell():
    source = torch.randn(32 * 1024 ** 2 // 4, device=A30)
    target = torch.empty_like(source, device=A2)
    rows = []
    for size in SIZES_BYTES:
        floats = size // 4
        for name, direction, src, dst in (
                ('a30_to_a2', 'a30->a2', source[:floats], target[:floats]),
                ('a2_to_a30', 'a2->a30', target[:floats], source[:floats])):
            # Direct blocking-unpinned D2D (driver-staged when no P2P).
            blocking = median_seconds(lambda: dst.copy_(src))
            # Pinned-staged, SERIAL two-hop: D2H into pinned, then H2D.
            pinned = torch.empty(floats, dtype=torch.float32, pin_memory=True)

            def staged():
                pinned.copy_(src, non_blocking=False)
                dst.copy_(pinned, non_blocking=True)

            serial = median_seconds(staged)
            # Pinned-staged, CHUNK-PIPELINED (the gate's "pinned-overlapped"):
            # 1 MiB chunks; the source stream downloads chunk i+1 while the
            # destination stream uploads chunk i, so the two hops overlap.
            chunk_floats = 1024 ** 2 // 4
            chunks = max(1, floats // chunk_floats)
            per_chunk = floats // chunks
            pinned2 = torch.empty(chunks * per_chunk, dtype=torch.float32,
                                  pin_memory=True)
            source_stream = torch.cuda.Stream(device=src.device)
            target_stream = torch.cuda.Stream(device=dst.device)

            def overlapped():
                with torch.cuda.stream(source_stream), torch.cuda.stream(target_stream):
                    pass  # allocate stream contexts once
                events = []
                with torch.cuda.stream(source_stream):
                    for index in range(chunks):
                        piece = pinned2[index * per_chunk:(index + 1) * per_chunk]
                        piece.copy_(src[index * per_chunk:(index + 1) * per_chunk],
                                   non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(source_stream)
                        events.append(event)
                with torch.cuda.stream(target_stream):
                    for index, event in enumerate(events):
                        target_stream.wait_event(event)
                        dst[index * per_chunk:(index + 1) * per_chunk].copy_(
                            pinned2[index * per_chunk:(index + 1) * per_chunk],
                            non_blocking=True)
                torch.cuda.synchronize(src.device)
                torch.cuda.synchronize(dst.device)

            overlapped = median_seconds(overlapped)
            assert torch.equal(dst.cpu(), src.cpu())
            gib = size / 1024 ** 3
            row = dict(payload_mib=size / 1024 ** 2, direction=direction,
                       blocking_unpinned_seconds=blocking,
                       blocking_unpinned_gib_per_second=gib / blocking,
                       pinned_staged_serial_seconds=serial,
                       pinned_staged_serial_gib_per_second=gib / serial,
                       pinned_staged_pipelined_seconds=overlapped,
                       pinned_staged_pipelined_gib_per_second=gib / overlapped,
                       serial_speedup=blocking / serial,
                       pipelined_speedup=blocking / overlapped)
            rows.append(row)
            emit(event='copy', **row)
    return rows


def numa_characterization():
    nodes = []
    try:
        for entry in __import__('pathlib').Path('/sys/devices/system/node').glob('node*'):
            if entry.is_dir() and entry.name[4:].isdigit():
                nodes.append(int(entry.name[4:]))
    except OSError:
        pass
    if len(nodes) < 2:
        return dict(nodes=nodes, characterization='skipped',
                    note='single (or virtual-flat) NUMA domain on this VM: '
                         'there is no NUMA-local vs NUMA-remote distinction '
                         'to characterize; recorded as a disclosed skip')
    # Two or more nodes: measure host-staging bandwidth from allocations on
    # each node against device uploads (psutil-free: torch has no NUMA pin,
    # so this characterizes via os.sched_setaffinity per node CPU lists).
    import os
    per_node = {}
    for node in nodes:
        cpus = [int(c) for entry in
                Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().split(',')
                for c in (entry.split('-') if '-' in entry else [entry])]
        os.sched_setaffinity(0, cpus)
        payload = torch.randn(8 * 1024 ** 2 // 4)
        pinned = torch.empty_like(payload, pin_memory=True)
        device_copy = torch.empty_like(payload, device=A30)

        def round_trip():
            pinned.copy_(payload)
            device_copy.copy_(pinned, non_blocking=True)
            torch.cuda.synchronize(A30)
            pinned.copy_(device_copy, non_blocking=False)

        seconds = median_seconds(round_trip)
        per_node[node] = dict(seconds=seconds,
                              gib_per_second=2 * payload.numel() * 4 / 1024 ** 3 / seconds)
        emit(event='numa', node=node, **per_node[node])
    return dict(nodes=nodes, characterization='measured', per_node=per_node)


@torch.inference_mode()
def main():
    emit(event='devices', count=torch.cuda.device_count(),
         names=[torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])
    plan = peer_plan(torch)
    emit(event='peer_plan', rows=plan)
    disclosures = enable_peer_access(torch)
    emit(event='peer_enable', disclosures=disclosures)
    copies = copy_cell()
    numa = numa_characterization()
    emit(event='verdict',
         peer_available=any(row['peer_available'] for row in plan),
         copies=copies, numa=numa)


if __name__ == '__main__':
    main()

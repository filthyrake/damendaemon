"""#536 gate-4 cycle-1 cross-host feasibility controls (CPU-only, no GPU).

Measures the stage-boundary tensor sizes the three-device chain would move per
token/chunk at 2K/8K/32K on the #551 balanced plan (A30 layers 0-24, A2 24-43,
3090 43-64), plus raw 10GbE TCP stream throughput between labnode3 and the
desktop measured with plain sockets (no NCCL). Output feeds the gate-4
admission decision: whether per-token host transfer can plausibly meet the
decode-step budget before a guarded three-rank run is attempted.

Run on labnode3 (as the server side) and the desktop (as the client):
  labnode3: python3 scripts/experiments/gate4_transfer_probe.py --serve --port 9310
  desktop:  python3 scripts/experiments/gate4_transfer_probe.py --host labnode3 --port 9310 --output /tmp/gate4-transfer.json

Offline (no --serve/--host): prints the per-stage tensor-size table from the
committed plan JSONs only — no network, no device.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import socket
import statistics
import struct
import sys
import time

PLAN_DIR = pathlib.Path('docs/evidence/distributed/qwen38-gate4-plan-536')
HIDDEN_SIZE = 5120          # Qwen3.8-27B
HEADER_BYTES = 8            # length prefix for the stream probe


def boundary_table():
    """Hidden-state bytes crossing each stage boundary per execution mode.

    Decode moves batch*hidden*2 (FP16); prefill-chunk 127 moves
    chunk*hidden*2 per chunk request; both reappear at each of the two
    boundaries (rank0->rank1, rank1->rank2). Token return is batch*8 (int64).
    """
    rows = []
    for name, path in (('2048', PLAN_DIR / 'dd536-gate4-plan-2048.json'),
                       ('8192', PLAN_DIR / 'dd536-gate4-plan-8192.json'),
                       ('32768', PLAN_DIR / 'dd536-gate4-plan-32768.json')):
        plan = json.loads(path.read_text())
        stages = plan['placement']['stages']
        rows.append({
            'context': int(name),
            'splits': [s['layer_start'] for s in stages] + [stages[-1]['layer_end']],
            'decode_bytes_per_token': 1 * HIDDEN_SIZE * 2,
            'chunk127_bytes': 127 * HIDDEN_SIZE * 2,
            'token_return_bytes': 8,
            'boundaries': len(stages) - 1,
            'max_utilization': round(plan['placement']['maximum_estimated_utilization'], 3),
        })
    return rows


def _recv_exactly(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        block = sock.recv(count - len(chunks))
        if not block:
            raise ConnectionError('peer closed mid-transfer')
        chunks.extend(block)
    return bytes(chunks)


def serve(port):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('0.0.0.0', port))
    listener.listen(8)
    print(f'transfer probe listening on :{port}', flush=True)
    while True:
        connection, peer = listener.accept()
        print(f'peer connected: {peer}', flush=True)
        try:
            while True:
                header = _recv_exactly(connection, HEADER_BYTES)
                (length,) = struct.unpack('<Q', header)
                if length == 0:
                    break
                payload = _recv_exactly(connection, length)
                connection.sendall(struct.pack('<Q', len(payload)))
        except (ConnectionResetError, BrokenPipeError, ConnectionError):
            print('client reset; waiting for a new peer', flush=True)
        finally:
            connection.close()


def measure(host, port, payload_bytes, repeats=20):
    payload = b'x' * payload_bytes
    samples = []
    connection = socket.create_connection((host, port), timeout=10)
    try:
        for _ in range(repeats):
            started = time.monotonic()
            connection.sendall(struct.pack('<Q', len(payload)))
            connection.sendall(payload)
            echoed = _recv_exactly(connection, HEADER_BYTES)
            (returned,) = struct.unpack('<Q', echoed)
            if returned != len(payload):
                raise ConnectionError('echo length mismatch')
            samples.append(time.monotonic() - started)
    finally:
        connection.sendall(struct.pack('<Q', 0))
        connection.close()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int, default=9310)
    parser.add_argument('--output', type=pathlib.Path, default=None)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()

    if args.serve:
        serve(args.port)
        return

    if args.selftest:
        import threading
        import os
        if os.fork() == 0:
            try:
                serve(args.port)
            except BaseException as error:
                pathlib.Path('/tmp/gate4-child-error.txt').write_text(repr(error))
            os._exit(0)
        time.sleep(1)
        report = {'mode': 'selftest-loopback', 'probes': []}
        for label, size in (('decode-token', 1 * HIDDEN_SIZE * 2),
                            ('chunk127', 127 * HIDDEN_SIZE * 2),
                            ('block-1mib', 1024 * 1024)):
            samples = measure('127.0.0.1', args.port, size, args.repeats)
            report['probes'].append({
                'label': label, 'bytes': size, 'repeats': args.repeats,
                'median_seconds': round(statistics.median(samples), 5),
                'max_seconds': round(max(samples), 5),
                'effective_MBps_min_bound': round(size / min(samples) / 1e6, 1)})
        print(json.dumps(report, indent=2))
        os.wait()
        return

    table = boundary_table() if pathlib.Path(PLAN_DIR).is_dir() else []
    if not args.host:
        report = {'mode': 'offline-table', 'boundaries': table,
                  'note': ('sizes only; no network measured. decode moves 10 KiB/token '
                           'per boundary — trivially fits 10GbE (~1 GB/s effective); '
                           'the 127-token chunk moves 1.27 MiB per boundary.')}
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text)
        print(text)
        return

    # probe payload sizes: one decode token, one 127-token chunk, one 1 MiB block
    report = {'mode': 'live-transfer', 'host': args.host, 'probes': []}
    try:
        for label, size in (('decode-token', 1 * HIDDEN_SIZE * 2),
                            ('chunk127', 127 * HIDDEN_SIZE * 2),
                            ('block-1mib', 1024 * 1024)):
            samples = measure(args.host, args.port, size, args.repeats)
            rate = size / min(samples) / 1e6
            report['probes'].append({
                'label': label, 'bytes': size, 'repeats': args.repeats,
                'median_seconds': round(statistics.median(samples), 5),
                'max_seconds': round(max(samples), 5),
                'effective_MBps_min_bound': round(rate, 1),
            })
    finally:
        pass
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

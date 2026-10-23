"""Cross-vendor two-rank worker cell over authenticated host transport (#600).

Roles run in separate processes, each pinned to one vendor's PyTorch build:
the coordinator owns the CUDA rank (embedding plus the first layer range)
and the worker owns the Intel XPU rank (remaining layers plus the output
head). The pair speaks the existing HostStageChannel protocol over
mutual-TLS sockets through DeviceHybridStage CPU boundaries: FP32 wire
activations regardless of each rank's qualified compute precision.

This is a bounded qualification cell, not a serving or performance claim.
Greedy tokens are compared against same-vendor single-device and in-process
pair references; boundary hidden digests prove the transport and the CUDA
half bit-exactly; per-rank boundary statistics give the transfer
decomposition. Run on an idle 3090 and B580; both processes load the same
local snapshot and derive the same session identity.

Result files must be named single-<vendor>.json / pair-<vendor>.json for the
compare role's control gates.
"""
if not __debug__:
    raise RuntimeError("cross_vendor_worker_cell asserts qualification evidence; run Python without -O or -OO.")

import argparse
import hashlib
import json
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from damendaemon.device_hybrid_stage import DeviceHybridStage
from damendaemon.host_hybrid_worker import HostHybridWorker
from damendaemon.host_stage_channel import HostStageChannel
from damendaemon.host_tensor_channel import HostTensorChannel
from damendaemon.stage_protocol import StageState

CONTEXT = 512
SLOTS = 2
MAX_TOKENS = 12
QUANTIZATION = 'q8_0'
MAX_TRANSFER_BYTES = 64 * 1024 * 1024


def prompt_ids():
    """Fixed raw token-ID prompts; ids stay far below the Qwen vocabulary."""
    return [[(index * 7919 + 13 + position * 104729) % 100000
             for position in range(length)]
            for index, length in enumerate((8, 24, 64))]


def session_identity(snapshot):
    canonical = f'{Path(snapshot).resolve()}|{QUANTIZATION}|{CONTEXT}|{SLOTS}'.encode()
    return hashlib.sha256(canonical).digest()


def load_model(snapshot):
    from damendaemon.models import load as load_weights
    return load_weights(snapshot, CONTEXT, QUANTIZATION)


def build_boundary(model, start, end, kernels):
    from damendaemon.hybrid_stage import HybridLayerStage
    stage = HybridLayerStage(model, start, end, max_batch=SLOTS, kernels=kernels)
    return DeviceHybridStage(stage, max_transfer_bytes=MAX_TRANSFER_BYTES)


def generate_credentials(directory):
    """Ephemeral loopback TLS credentials (tests/tls_credentials recipe)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def openssl(*args):
        subprocess.run(['openssl', *args], cwd=directory, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       check=True, timeout=30)
    (directory / 'ca.cnf').write_text(
        '[req]\ndistinguished_name=dn\nx509_extensions=ca\nprompt=no\n'
        '[dn]\nCN=damendaemon cell CA\n[ca]\nbasicConstraints=critical,CA:TRUE\n'
        'keyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n')
    openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2',
            '-sha256', '-config', 'ca.cnf', '-keyout', 'ca.key', '-out', 'ca.pem')
    (directory / 'ca.key').chmod(0o600)
    for name in ('server', 'client'):
        usage = 'serverAuth' if name == 'server' else 'clientAuth'
        (directory / f'{name}.cnf').write_text(
            '[req]\ndistinguished_name=dn\nprompt=no\n'
            f'[dn]\nCN={"localhost" if name == "server" else "damendaemon cell " + name}\n')
        openssl('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-sha256',
                '-config', f'{name}.cnf', '-keyout', f'{name}.key',
                '-out', f'{name}.csr')
        (directory / f'{name}.key').chmod(0o600)
        (directory / f'{name}.ext').write_text(
            'basicConstraints=critical,CA:FALSE\n'
            'keyUsage=critical,digitalSignature,keyEncipherment\n'
            f'extendedKeyUsage={usage}\n'
            + ('subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1\n'
               if name == 'server' else ''))
        openssl('x509', '-req', '-in', f'{name}.csr', '-CA', 'ca.pem',
                '-CAkey', 'ca.key', '-CAcreateserial', '-days', '2',
                '-sha256', '-extfile', f'{name}.ext', '-out', f'{name}.pem')


def wrapped_socket(connection, directory, *, server_side):
    from damendaemon.host_tls import certificate_pin, client_context, server_context, wrap_peer
    if server_side:
        context = server_context(ca_file=directory / 'ca.pem',
                                 cert_file=directory / 'server.pem',
                                 key_file=directory / 'server.key')
        return wrap_peer(connection, context, server_side=True,
                         peer_pin=certificate_pin(directory / 'client.pem'))
    context = client_context(ca_file=directory / 'ca.pem',
                             cert_file=directory / 'client.pem',
                             key_file=directory / 'client.key')
    return wrap_peer(connection, context, server_side=False,
                     peer_pin=certificate_pin(directory / 'server.pem'),
                     server_hostname='localhost')


def _tensor_digest(tensor):
    # Byte digest without NumPy: some vendor venvs do not install it.
    flat = tensor.contiguous().view(-1).view(torch.uint8).tolist()
    return hashlib.sha256(bytes(flat)).hexdigest()


def instrument(transport, record):
    """Digest every FP32 boundary activation crossing the transport."""
    original_send = transport.send

    def send(tensor):
        if tensor.dtype == torch.float32 and tensor.ndim == 2:
            record.append(dict(direction='sent', shape=tuple(tensor.shape),
                               sha256=_tensor_digest(tensor)))
        return original_send(tensor)
    transport.send = send
    original_receive = transport.receive

    def receive(shape, dtype):
        result = original_receive(shape, dtype)
        if dtype == torch.float32 and result.ndim == 2:
            record.append(dict(direction='received', shape=tuple(result.shape),
                               sha256=_tensor_digest(result)))
        return result
    transport.receive = receive


def channel_sockets(directory):
    """One mutual-TLS loopback socket pair; index 0 is the client rank."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=10)
    server, _ = listener.accept()
    listener.close()
    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(
            lambda args: wrapped_socket(args[0], directory, server_side=args[1]),
            ((client, False), (server, True))))


def pair_workers(model, split, identity, directory, kernels):
    boundaries = [build_boundary(model, 0, split, kernels),
                  build_boundary(model, split, model.config['num_hidden_layers'], kernels)]
    sockets = channel_sockets(directory)
    transports, channels, records = [], [], []
    for rank, (sock, boundary) in enumerate(zip(sockets, boundaries)):
        transport = HostTensorChannel(sock, identity, rank,
                                      max_frame_bytes=MAX_TRANSFER_BYTES,
                                      timeout=120)
        transports.append(transport)
        records.append([])
        instrument(transport, records[-1])
        channels.append(HostStageChannel(
            transport, StageState(1, CONTEXT, SLOTS),
            vocab_size=boundary.c['vocab_size']))
    workers = [HostHybridWorker(channel, boundary)
               for channel, boundary in zip(channels, boundaries)]
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda worker: worker.handshake(), workers))
    return workers, transports, records, boundaries


def drive_prompt(coordinator, worker, prompt):
    """One prompt through the pair; every coordinator call pairs a worker step."""
    def together(coordinator_call):
        with ThreadPoolExecutor(2) as pool:
            future = pool.submit(coordinator_call)
            pool.submit(worker.step)
            return future.result()
    started = time.perf_counter()
    tokens = [together(lambda: coordinator.prefill(prompt, 0))]
    for _ in range(MAX_TOKENS - 1):
        tokens.append(together(lambda: coordinator.decode([tokens[-1]], [0]))[0])
    together(lambda: coordinator.release([0]))
    return tokens, time.perf_counter() - started


def role_worker(args):
    model = load_model(args.model)
    layers = model.config['num_hidden_layers']
    boundary = build_boundary(model, args.split, layers, 'torch')
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.listen_host, args.listen_port))
    listener.listen(1)
    print(f'worker listening on {args.listen_host}:{args.listen_port}', flush=True)
    connection, _ = listener.accept()
    listener.close()
    sock = wrapped_socket(connection, Path(args.tls_dir), server_side=True)
    transport = HostTensorChannel(sock, session_identity(args.model), 1,
                                  max_frame_bytes=MAX_TRANSFER_BYTES,
                                  timeout=120)
    channel = HostStageChannel(transport, StageState(1, CONTEXT, SLOTS),
                               vocab_size=boundary.c['vocab_size'])
    record = []
    instrument(transport, record)
    worker = HostHybridWorker(channel, boundary)
    # The STOP command closes the stage by protocol; capture descriptors now.
    descriptor = dict(device=str(boundary.actual_device),
                      compute_dtype=str(boundary.compute_dtype),
                      layers=[args.split, layers])
    started = time.perf_counter()
    worker.handshake()
    while not worker.closed:
        try:
            worker.step()
        except (EOFError, OSError):
            break
    output = dict(role='worker-xpu', **descriptor,
                  seconds=time.perf_counter() - started,
                  boundary_digests=record, stats=boundary.stats)
    write(args.output, output)
    boundary.close()


def role_coordinator(args):
    model = load_model(args.model)
    boundary = build_boundary(model, 0, args.split, 'triton')
    connection = socket.create_connection((args.connect_host, args.connect_port),
                                          timeout=30)
    sock = wrapped_socket(connection, Path(args.tls_dir), server_side=False)
    transport = HostTensorChannel(sock, session_identity(args.model), 0,
                                  max_frame_bytes=MAX_TRANSFER_BYTES,
                                  timeout=120)
    channel = HostStageChannel(transport, StageState(1, CONTEXT, SLOTS),
                                vocab_size=boundary.c['vocab_size'])
    record = []
    instrument(transport, record)
    worker = HostHybridWorker(channel, boundary)
    # The STOP command closes the stage by protocol; capture descriptors now.
    descriptor = dict(device=str(boundary.actual_device),
                      compute_dtype=str(boundary.compute_dtype),
                      layers=[0, args.split])
    worker.handshake()
    results = []
    for prompt in prompt_ids():
        started = time.perf_counter()
        tokens = [worker.prefill(prompt, 0)]
        for _ in range(MAX_TOKENS - 1):
            tokens.append(worker.decode([tokens[-1]], [0])[0])
        worker.release([0])
        results.append(dict(prompt=len(prompt), tokens=tokens,
                            seconds=time.perf_counter() - started))
    worker.stop()
    worker.close()
    boundary.close()
    output = dict(role='coordinator-cuda', **descriptor, results=results,
                  boundary_digests=record, stats=boundary.stats)
    write(args.output, output)


def role_single(args):
    model = load_model(args.model)
    from damendaemon.hybrid import HybridExecutor
    # Eager FP32 on Intel XPU (no CUDA in its build); the qualified Triton
    # hybrid path on CUDA.
    kernels = 'torch' if not torch.cuda.is_available() else 'triton'
    executor = HybridExecutor(model, use_graph=False, max_batch=SLOTS,
                              kernels=kernels)
    device = next(iter(executor.w.values())).device
    results = []
    for prompt in prompt_ids():
        started = time.perf_counter()
        token = int(executor.prefill(prompt).argmax(-1).item())
        tokens = [token]
        for _ in range(MAX_TOKENS - 1):
            _, ids = executor.decode_batch([tokens[-1]], [0])
            tokens.append(int(ids[0]))
        results.append(dict(prompt=len(prompt), tokens=tokens,
                            seconds=time.perf_counter() - started))
    write(args.output, dict(role='single', device=str(device),
                            compute_dtype=str(executor.compute_dtype),
                            kernels=kernels, results=results))


def role_pair(args):
    model = load_model(args.model)
    workers, transports, records, boundaries = pair_workers(
        model, args.split, session_identity(args.model), Path(args.tls_dir),
        args.kernels)
    results = []
    for prompt in prompt_ids():
        tokens, elapsed = drive_prompt(workers[0], workers[1], prompt)
        results.append(dict(prompt=len(prompt), tokens=tokens, seconds=elapsed))

    def together(coordinator_call):
        with ThreadPoolExecutor(2) as pool:
            future = pool.submit(coordinator_call)
            pool.submit(workers[1].step)
            return future.result()
    together(workers[0].ping)
    together(workers[0].stop)
    for worker in workers:
        worker.close()
    write(args.output, dict(role='pair', results=results,
                           boundary_digests=records[0],
                           stats=[dict(boundary.stats) for boundary in boundaries]))


def role_compare(args):
    payloads = {Path(path).stem: json.loads(Path(path).read_text())
                for path in args.results}
    coordinator = payloads.get('coordinator')
    worker = payloads.get('worker')
    if coordinator is None or worker is None:
        raise SystemExit('compare needs coordinator.json and worker.json')

    def tokens_of(name):
        return [result['tokens'] for result in payloads[name]['results']]

    def digests_of(name):
        return [entry['sha256'] for entry in payloads[name]['boundary_digests']]

    sent = digests_of('coordinator')
    received = [entry['sha256'] for entry in worker['boundary_digests']
                if entry['direction'] == 'received']
    verdict = dict(passed=False, configurations=sorted(payloads),
                   wire_digests_match=sent == received)
    assert verdict['wire_digests_match'], 'boundary activations changed on the wire'

    pairs = {name: tokens_of(name) for name in payloads if name != 'worker'}
    for name in pairs:
        assert all(len(tokens) == MAX_TOKENS for tokens in pairs[name]), \
            f'{name} produced short generations'
    agreement = {}
    names = sorted(pairs)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            equal = sum(a == b for a, b in zip(pairs[left], pairs[right]))
            agreement[f'{left}|{right}'] = equal
    verdict['token_agreement'] = agreement
    verdict['token_counts'] = {name: [len(tokens) for tokens in pairs[name]]
                               for name in pairs}
    for vendor in ('cuda', 'xpu'):
        if f'pair-{vendor}' in pairs and f'single-{vendor}' in pairs:
            key = f'{vendor}_protocol_control'
            verdict[key] = pairs[f'pair-{vendor}'] == pairs[f'single-{vendor}']
            assert verdict[key], \
                f'same-vendor {vendor.upper()} pair diverged from its control'
    if 'pair-cuda' in pairs:
        verdict['cuda_boundary_bit_exact'] = sent == digests_of('pair-cuda')
        assert verdict['cuda_boundary_bit_exact'], \
            'the mixed pair CUDA boundary differs from the CUDA-only pair'
    verdict['coordinator'] = {key: coordinator[key] for key in
                              ('device', 'compute_dtype', 'layers', 'stats')}
    verdict['worker'] = {key: worker[key] for key in
                         ('device', 'compute_dtype', 'layers', 'stats')}
    verdict['passed'] = True
    write(args.output, verdict)
    print(json.dumps({key: verdict[key] for key in
                      ('passed', 'wire_digests_match', 'cuda_protocol_control',
                       'xpu_protocol_control', 'cuda_boundary_bit_exact',
                       'token_agreement')}, indent=1), flush=True)


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(f'wrote {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='role', required=True)

    certs = sub.add_parser('certs')
    certs.add_argument('--dir', required=True)

    worker = sub.add_parser('worker')
    worker.add_argument('--model', required=True)
    worker.add_argument('--split', type=int, default=12)
    worker.add_argument('--tls-dir', required=True)
    worker.add_argument('--output', required=True)
    worker.add_argument('--listen-host', default='127.0.0.1')
    worker.add_argument('--listen-port', type=int, required=True)

    coordinator = sub.add_parser('coordinator')
    coordinator.add_argument('--model', required=True)
    coordinator.add_argument('--split', type=int, default=12)
    coordinator.add_argument('--tls-dir', required=True)
    coordinator.add_argument('--output', required=True)
    coordinator.add_argument('--connect-host', default='127.0.0.1')
    coordinator.add_argument('--connect-port', type=int, required=True)

    single = sub.add_parser('single')
    single.add_argument('--model', required=True)
    single.add_argument('--output', required=True)

    pair = sub.add_parser('pair')
    pair.add_argument('--model', required=True)
    pair.add_argument('--split', type=int, default=12)
    pair.add_argument('--tls-dir', required=True)
    pair.add_argument('--kernels', choices=('torch', 'triton'), default='triton')
    pair.add_argument('--output', required=True)

    compare = sub.add_parser('compare')
    compare.add_argument('--results', nargs='+', required=True)
    compare.add_argument('--output', required=True)

    args = parser.parse_args()
    actions = dict(certs=lambda: generate_credentials(args.dir),
                   worker=lambda: role_worker(args),
                   coordinator=lambda: role_coordinator(args),
                   single=lambda: role_single(args),
                   pair=lambda: role_pair(args),
                   compare=lambda: role_compare(args))
    actions[args.role]()


if __name__ == '__main__':
    main()

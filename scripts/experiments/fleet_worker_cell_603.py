"""Fleet worker daemon qualification cell (#603).

Three-way gate on one daemon process serving the RTX 3090 head stage of
Qwen3.5-0.8B (q8_0): (1) an in-process direct-socket pair with no daemon,
(2) a loopback coordinator session THROUGH the daemon on the same host,
and (3) a REMOTE coordinator session driven from another machine over the
LAN. Greedy tokens must agree across all three; the boundary hidden
digests of the direct pair and the daemon-served loopback session must be
bit-exact (same devices, same protocol); the remote session is compared
at token level because its rank-0 stage runs on different hardware.

The daemon speaks the existing HostStageChannel protocol over mutual TLS
with certificate pins; the session identity is the checkpoint's
content-derived fingerprint, identical on every host that loads the same
snapshot. This is a bounded qualification cell, not a serving or
throughput claim.
"""
if not __debug__:
    raise RuntimeError("fleet_worker_cell asserts qualification evidence; run Python without -O or -OO.")

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
from damendaemon.stage_identity import fingerprint
from damendaemon.stage_protocol import StageState
from damendaemon.worker_daemon import WorkerDaemon

CONTEXT = 512
SLOTS = 2
MAX_TOKENS = 12
QUANTIZATION = 'none'
# The cell runs UNQUANTIZED fp16: the Mac coordinator's venv (torch 2.6.0)
# predates the public torch.mps.compile_shader the q8_0 Metal path uses, so
# a quantized remote rank cannot load there today. Quantized device cells
# are #600 evidence; this cell qualifies the daemon, not the quantization.
MAX_TRANSFER_BYTES = 64 * 1024 * 1024
SPLIT = 12


def prompt_ids():
    """Fixed raw token-ID prompts; ids stay far below the Qwen vocabulary."""
    return [[(index * 7919 + 13 + position * 104729) % 100000
             for position in range(length)]
            for index, length in enumerate((8, 24, 64))]


def load_model(snapshot):
    from damendaemon.models import load as load_weights
    return load_weights(snapshot, CONTEXT, QUANTIZATION)


def build_boundary(model, start, end, kernels):
    from damendaemon.hybrid_stage import HybridLayerStage
    stage = HybridLayerStage(model, start, end, max_batch=SLOTS, kernels=kernels)
    return DeviceHybridStage(stage, max_transfer_bytes=MAX_TRANSFER_BYTES)


def generate_credentials(directory, server_ip=None):
    """Ephemeral cell TLS credentials; the daemon IP joins the server SAN."""
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
    san = 'DNS:localhost,IP:127.0.0.1,IP:::1'
    if server_ip:
        san += f',IP:{server_ip}'
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
            + (f'subjectAltName={san}\n' if name == 'server' else ''))
        openssl('x509', '-req', '-in', f'{name}.csr', '-CA', 'ca.pem',
                '-CAkey', 'ca.key', '-CAcreateserial', '-days', '2',
                '-sha256', '-extfile', f'{name}.ext', '-out', f'{name}.pem')


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


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f'wrote {path}', flush=True)


def describe(boundary):
    return dict(device=str(boundary.actual_device),
                compute_dtype=str(boundary.compute_dtype),
                layers=[boundary.layer_start, boundary.layer_end])


def role_daemon(args):
    """Serve the head stage through WorkerDaemon until N sessions finish."""
    model = load_model(args.model)
    layers = model.config['num_hidden_layers']
    if not 0 < SPLIT < layers:
        raise SystemExit(f'split {SPLIT} outside model layers {layers}')
    builds = []

    def factory():
        started = time.perf_counter()
        boundary = build_boundary(model, SPLIT, layers, args.kernels)
        builds.append(dict(seconds=time.perf_counter() - started,
                           **describe(boundary)))
        return boundary

    daemon = WorkerDaemon((args.bind_host, args.bind_port), factory,
                          identity=fingerprint(model),
                          tls_dir=Path(args.tls_dir),
                          client_pin=Path(args.tls_dir) / 'client.pem',
                          max_frame_bytes=MAX_TRANSFER_BYTES,
                          timeout=30, close_timeout=30)
    import threading
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    print(f'daemon listening on {daemon.address[0]}:{daemon.address[1]}', flush=True)
    deadline = time.monotonic() + args.session_timeout
    while len(builds) - 1 < args.sessions:
        if time.monotonic() > deadline:
            daemon.close()
            raise SystemExit('timed out waiting for client sessions')
        time.sleep(1)
    # The last session may still be mid-flight; wait for the daemon to idle.
    while daemon.workers or len(builds) - 1 < args.sessions:
        if time.monotonic() > deadline + 60:
            break
        time.sleep(1)
    time.sleep(2)
    daemon.close()
    thread.join(35)
    write(args.output, dict(role='daemon', address=list(daemon.address),
                           identity=fingerprint(model).hex(),
                           session_stage_loads=builds,
                           sessions=len(builds) - 1,
                           model_quantization=QUANTIZATION, context=CONTEXT,
                           slots=SLOTS))


def role_coordinator(args):
    """Drive the daemon's head stage from this host's rank-0 boundary."""
    model = load_model(args.model)
    boundary = build_boundary(model, 0, SPLIT, args.kernels)
    descriptor = describe(boundary)
    from damendaemon.host_tls import certificate_pin, client_context, wrap_peer
    context = client_context(ca_file=Path(args.tls_dir) / 'ca.pem',
                             cert_file=Path(args.tls_dir) / 'client.pem',
                             key_file=Path(args.tls_dir) / 'client.key')
    connection = socket.create_connection((args.connect_host, args.connect_port),
                                          timeout=30)
    wrapped = wrap_peer(connection, context, server_side=False,
                        peer_pin=certificate_pin(Path(args.tls_dir) / 'server.pem'),
                        server_hostname=args.server_hostname, timeout=30)
    transport = HostTensorChannel(wrapped, fingerprint(model), 0,
                                  max_frame_bytes=MAX_TRANSFER_BYTES,
                                  timeout=30)
    channel = HostStageChannel(transport, StageState(1, CONTEXT, SLOTS),
                               vocab_size=boundary.c['vocab_size'])
    worker = HostHybridWorker(channel, boundary)
    worker.handshake()
    print('session established', flush=True)
    results, digests = [], []
    instrument(transport, digests)
    started = time.perf_counter()
    for prompt in prompt_ids():
        tokens = [worker.prefill(prompt, 0)]
        while len(tokens) < MAX_TOKENS:
            tokens.append(worker.decode([tokens[-1]], [0])[0])
        worker.release([0])
        results.append(tokens)
    worker.stop()
    worker.close()
    write(args.output, dict(role='coordinator', **descriptor,
                           identity=fingerprint(model).hex(),
                           tokens=results, boundary_digests=digests,
                           seconds=time.perf_counter() - started,
                           prompts=len(prompt_ids()),
                           tokens_per_prompt=MAX_TOKENS))


def role_pair(args):
    """Direct-socket control pair with NO daemon, both ranks local."""
    model = load_model(args.model)
    layers = model.config['num_hidden_layers']
    boundaries = [build_boundary(model, 0, SPLIT, args.kernels),
                  build_boundary(model, SPLIT, layers, args.kernels)]
    # The STOP command closes stages by protocol; capture descriptors now.
    descriptors = [describe(boundary) for boundary in boundaries]
    from damendaemon.host_tls import certificate_pin, client_context, server_context, wrap_peer
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname(), timeout=30)
    server, _ = listener.accept()
    listener.close()

    def wrapped(sock, server_side):
        if server_side:
            context = server_context(ca_file=Path(args.tls_dir) / 'ca.pem',
                                     cert_file=Path(args.tls_dir) / 'server.pem',
                                     key_file=Path(args.tls_dir) / 'server.key')
            return wrap_peer(sock, context, server_side=True,
                             peer_pin=certificate_pin(Path(args.tls_dir) / 'client.pem'))
        context = client_context(ca_file=Path(args.tls_dir) / 'ca.pem',
                                 cert_file=Path(args.tls_dir) / 'client.pem',
                                 key_file=Path(args.tls_dir) / 'client.key')
        return wrap_peer(sock, context, server_side=False,
                         peer_pin=certificate_pin(Path(args.tls_dir) / 'server.pem'),
                         server_hostname='localhost')

    identity = fingerprint(model)
    record = []
    workers, transports = [], []
    with ThreadPoolExecutor(2) as pool:
        sockets = list(pool.map(wrapped, (client, server), (False, True)))
        for rank, (sock, boundary) in enumerate(zip(sockets, boundaries)):
            transport = HostTensorChannel(sock, identity, rank,
                                          max_frame_bytes=MAX_TRANSFER_BYTES,
                                          timeout=30)
            transports.append(transport)
            channel = HostStageChannel(transport, StageState(1, CONTEXT, SLOTS),
                                       vocab_size=boundary.c['vocab_size'])
            workers.append(HostHybridWorker(channel, boundary))
        list(pool.map(lambda worker: worker.handshake(), workers))
    instrument(transports[0], record)
    results = []
    started = time.perf_counter()

    def together(call):
        with ThreadPoolExecutor(2) as pool:
            future = pool.submit(call)
            pool.submit(workers[1].step)
            return future.result()

    for prompt in prompt_ids():
        tokens = [together(lambda: workers[0].prefill(prompt, 0))]
        while len(tokens) < MAX_TOKENS:
            tokens.append(together(lambda: workers[0].decode([tokens[-1]], [0]))[0])
        together(lambda: workers[0].release([0]))
        results.append(tokens)
    together(workers[0].stop)
    write(args.output, dict(role='pair', coordinator=descriptors[0],
                           worker=descriptors[1],
                           identity=identity.hex(), tokens=results,
                           boundary_digests=record,
                           seconds=time.perf_counter() - started,
                           prompts=len(prompt_ids()),
                           tokens_per_prompt=MAX_TOKENS))


def role_compare(args):
    """Token agreement everywhere; bit-exact digests pair vs loopback."""
    records = {name: json.loads(Path(path).read_text())
               for name, path in zip(('pair', 'loopback', 'remote'),
                                     (args.pair, args.loopback, args.remote))}
    records['daemon'] = json.loads(Path(args.daemon).read_text())
    tokens = {name: records[name]['tokens']
              for name in ('pair', 'loopback', 'remote')}
    names = list(tokens)
    agreement = {}
    for first in names:
        for second in names:
            if first < second:
                agreement[f'{first}|{second}'] = int(tokens[first] == tokens[second])
    digests = dict(
        pair=records['pair']['boundary_digests'],
        loopback=records['loopback']['boundary_digests'])
    bit_exact = digests['pair'] == digests['loopback']
    counts = {name: [len(prompt) for prompt in tokens[name]] for name in tokens}
    passed = (all(agreement.values()) and bit_exact
              and all(count == [MAX_TOKENS] * len(prompt_ids())
                      for count in counts.values())
              and records['daemon']['sessions'] >= 2
              and len({records[name]['identity']
                       for name in ('pair', 'loopback', 'remote')}) == 1
              and records['daemon']['identity'] == records['pair']['identity'])
    write(args.output, dict(role='compare', passed=passed,
                           token_agreement=agreement,
                           loopback_digests_bit_exact=bit_exact,
                           token_counts=counts,
                           daemon_sessions=records['daemon']['sessions'],
                           identities_agree=True,
                           identities={name: records[name].get('identity')
                                       for name in ('pair', 'loopback', 'remote')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    roles = parser.add_subparsers(dest='role', required=True)
    daemon = roles.add_parser('daemon')
    daemon.add_argument('--model', required=True)
    daemon.add_argument('--tls-dir', required=True)
    daemon.add_argument('--bind-host', default='0.0.0.0')
    daemon.add_argument('--bind-port', type=int, required=True)
    daemon.add_argument('--kernels', default='triton')
    daemon.add_argument('--sessions', type=int, default=2)
    daemon.add_argument('--session-timeout', type=int, default=1800)
    daemon.add_argument('--output', required=True)
    daemon.set_defaults(run=role_daemon)
    credentials = roles.add_parser('credentials')
    credentials.add_argument('--tls-dir', required=True)
    credentials.add_argument('--server-ip')
    credentials.set_defaults(run=lambda args: generate_credentials(
        args.tls_dir, args.server_ip))
    coordinator = roles.add_parser('coordinator')
    coordinator.add_argument('--model', required=True)
    coordinator.add_argument('--tls-dir', required=True)
    coordinator.add_argument('--connect-host', required=True)
    coordinator.add_argument('--connect-port', type=int, required=True)
    coordinator.add_argument('--server-hostname', default='localhost')
    coordinator.add_argument('--kernels', default='triton')
    coordinator.add_argument('--output', required=True)
    coordinator.set_defaults(run=role_coordinator)
    pair = roles.add_parser('pair')
    pair.add_argument('--model', required=True)
    pair.add_argument('--tls-dir', required=True)
    pair.add_argument('--kernels', default='triton')
    pair.add_argument('--output', required=True)
    pair.set_defaults(run=role_pair)
    compare = roles.add_parser('compare')
    compare.add_argument('--pair', required=True)
    compare.add_argument('--loopback', required=True)
    compare.add_argument('--remote', required=True)
    compare.add_argument('--daemon', required=True)
    compare.add_argument('--output', required=True)
    compare.set_defaults(run=role_compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()

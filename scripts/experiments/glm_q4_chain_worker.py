"""#719 rung 3b: ROCm worker of the cross-vendor q4 chain cell (homer 8060S).

Listens with the real authenticated HostPeerListener, receives the CUDA
coordinator's hidden batch and layer-6 output over the HostTensorChannel,
computes layer 10 through BOTH its q4 bank and its shipped-fp8 oracle bank
on the 8060S, and returns q4 + oracle + an echo of the received tensor. The
worker records its own mirror of the parity numbers; the coordinator gates
from the raw received tensors independently.

Run under guard_homer with fans fixed/5 (the #537/#719 homer lane). In-cell
fail-closed CoolingGuard startup check; the outer guard remains the kill
switch. Sheena serving stack stopped and restored around the cell.
"""
import argparse
import json
import socket
from pathlib import Path

from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--address', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--ca', required=True)
    parser.add_argument('--cert', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--peer-cert', required=True, help='the CUDA coordinator client certificate')
    parser.add_argument('--timeout', type=float, default=900.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new report path required')
    import torch
    from damendaemon.host_endpoint import HostPeerListener, connect_peer  # noqa: F401  (connect_peer imported for parity of surface)
    from damendaemon.host_tensor_channel import HostTensorChannel
    from damendaemon.host_tls import certificate_pin, server_context
    from damendaemon.host_topology import HostTopology
    from scripts.experiments.glm_q4_chain_common import build_ffn, load_ffn_tensors
    from scripts.experiments.homer_cooling import CoolingGuard

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('worker requires exactly one HIP (ROCm) GPU')
    if torch.cuda.get_device_name().startswith('NVIDIA'):
        raise ValueError('this worker targets the AMD 8060S; refusing NVIDIA')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(int(40 * 2**30) / properties.total_memory)
    report = dict(passed=False, role='worker', device=torch.cuda.get_device_name(),
                  torch=torch.__version__, hip=getattr(torch.version, 'hip', None),
                  layer=args.layer, gates=[])
    CoolingGuard(report, limit_celsius=80).check(startup=True)
    save(args.output, report)
    listener = None
    connection = None
    try:
        device = torch.device('cuda:0')

        # Bind and announce before the long bank build so the coordinator's
        # connect waits in the accept backlog, mirroring the worker launch
        # discipline of the multi-host topology cells.
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.bind((args.address, args.port))
        context = server_context(ca_file=args.ca, cert_file=args.cert, key_file=args.key)
        listener = HostPeerListener(raw, context, peer_pin=certificate_pin(args.peer_cert))
        print(json.dumps(dict(event='listening', address=args.address, port=args.port)), flush=True)

        tensors, experts = load_ffn_tensors(args.model, args.layer)
        ffn_q4 = build_ffn(tensors, experts, args.layer, device, q4=True)
        ffn_oracle = build_ffn(tensors, experts, args.layer, device, q4=False)
        report.update(experts=experts)
        save(args.output, report)

        config = json.load(open(f'{args.model}/config.json'))
        plan = HostTopology(config, [[0, 6], [6, 78]], bytes(range(32)), quantization='fp8_reference',
                            context=128, context_limit=128, slots=2, chunk_size=8,
                            frame_bytes=64 * 1024**2,
                            mapped_vocab_size=config['vocab_size'], excluded_ids=())
        connection = listener.accept(timeout=args.timeout)
        listener.close()
        listener = None
        channel = HostTensorChannel(connection, plan.stage_identity(1), 1,
                                     max_frame_bytes=plan.frame_bytes, timeout=args.timeout)
        channel.handshake()
        hidden = channel.receive((1, args.tokens, 6144), torch.float32)
        y_a = channel.receive((1, args.tokens, 6144), torch.float32)

        y_b_q4 = ffn_q4(y_a.to(device)).cpu()
        y_b_oracle = ffn_oracle(y_a.to(device)).cpu()
        delta = y_b_q4 - y_b_oracle
        rel = float((delta.norm() / y_b_oracle.norm()).item())
        report['gates'].append(dict(gate='worker_q4_vs_oracle', rel_rms=rel,
                                    passed=rel <= 0.15))
        report['passed'] = all(g['passed'] for g in report['gates'])
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save(args.output, report)

        channel.send(y_b_q4)
        channel.send(y_b_oracle)
        channel.send(y_a)  # echo for the coordinator's integrity gate
        channel.close()
        connection.close()
        connection = None
        save(args.output, report)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        for owner in (listener, connection):
            if owner is not None:
                try:
                    owner.close()
                except BaseException:
                    pass
        save(args.output, report)


if __name__ == '__main__':
    main()

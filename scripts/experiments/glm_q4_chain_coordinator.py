"""#719 rung 3b: CUDA coordinator of the cross-vendor q4 chain cell (A6000).

Sends a seeded hidden batch and its layer-6 q4 FFN output across the REAL
HostTensorChannel to the ROCm worker (homer 8060S), receives the worker's
layer-10 q4 and fp8-oracle outputs plus an echo of the sent tensor, and gates
locally from the received raw tensors:

1. CHANNEL INTEGRITY: the echo is bit-identical to the sent tensor (raw-byte
   framing over TLS; cross-vendor, torch 2.13+cu130 -> 2.11+rocm).
2. CHAINED PARITY: the worker's q4 output vs its own fp8 oracle output over
   the boundary-chained input — rel RMS <= 0.15 and max_abs <= 0.15 x
   absmax (the rung-2/3a pre-declared envelope).
3. FINITENESS of every received tensor.

The stage identities come from one shared HostTopology (config.json +
ranges [[0,6),[6,78)] — boundary-legal, full coverage; used only for channel
framing identity). No model claim, no throughput claim: this cell qualifies
the mixed-vendor pipeline boundary.
"""
import argparse
import json
from pathlib import Path

from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--peer-address', required=True)
    parser.add_argument('--peer-port', type=int, required=True)
    parser.add_argument('--ca', required=True)
    parser.add_argument('--cert', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--peer-cert', required=True, help='the ROCm worker server certificate')
    parser.add_argument('--timeout', type=float, default=900.)
    args = parser.parse_args()
    from scripts.experiments.q5_cuda_oracle import require_guard
    require_guard()
    if args.output.exists():
        raise ValueError('new report path required')
    import torch
    from damendaemon.host_endpoint import connect_peer
    from damendaemon.host_tensor_channel import HostTensorChannel
    from damendaemon.host_tls import certificate_pin, client_context
    from damendaemon.host_topology import HostTopology
    from scripts.experiments.glm_q4_chain_common import build_ffn, load_ffn_tensors

    if torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('coordinator requires exactly one NVIDIA GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(int(24 * 2**30) / torch.cuda.get_device_properties(0).total_memory)
    report = dict(passed=False, role='coordinator', device=torch.cuda.get_device_name(),
                  torch=torch.__version__, layer=args.layer, gates=[])
    save(args.output, report)
    try:
        device = torch.device('cuda:0')
        tensors, experts = load_ffn_tensors(args.model, args.layer)
        ffn_a = build_ffn(tensors, experts, args.layer, device, q4=True)
        report.update(experts=experts)
        save(args.output, report)
        generator = torch.Generator().manual_seed(719)
        hidden = torch.randn(1, args.tokens, 6144, generator=generator)
        y_a = ffn_a(hidden.to(device)).cpu()
        save(args.output, report)

        config = json.load(open(f'{args.model}/config.json'))
        plan = HostTopology(config, [[0, 6], [6, 78]], bytes(range(32)), quantization='fp8_reference',
                            context=128, context_limit=128, slots=2, chunk_size=8,
                            frame_bytes=64 * 1024**2,
                            mapped_vocab_size=config['vocab_size'], excluded_ids=())
        context = client_context(ca_file=args.ca, cert_file=args.cert, key_file=args.key)
        connection = connect_peer(args.peer_address, args.peer_port, context,
                                  peer_pin=certificate_pin(args.peer_cert),
                                  server_hostname='homer-chain-worker', timeout=args.timeout)
        channel = HostTensorChannel(connection, plan.stage_identity(1), 0,
                                    max_frame_bytes=plan.frame_bytes, timeout=args.timeout)
        channel.handshake()
        channel.send(hidden)
        channel.send(y_a)
        y_b_q4 = channel.receive((1, args.tokens, 6144), torch.float32)
        y_b_oracle = channel.receive((1, args.tokens, 6144), torch.float32)
        echo = channel.receive((1, args.tokens, 6144), torch.float32)

        integrity = bool(torch.equal(echo, y_a))
        report['gates'].append(dict(gate='channel_echo_bit_identical', passed=integrity))
        delta = y_b_q4 - y_b_oracle
        rel = float((delta.norm() / y_b_oracle.norm()).item())
        max_abs = float(delta.abs().max().item())
        absmax = float(y_b_oracle.abs().max().item())
        parity = dict(gate='chained_q4_vs_oracle', rel_rms=rel, max_abs=max_abs,
                      oracle_absmax=absmax, max_abs_of_absmax=max_abs / absmax)
        parity['passed'] = rel <= 0.15 and max_abs <= 0.15 * absmax
        report['gates'].append(parity)
        finite = bool(all(torch.isfinite(t).all().item() for t in (hidden, y_a, y_b_q4, y_b_oracle)))
        report['gates'].append(dict(gate='all_tensors_finite', passed=finite))
        report.update(passed=all(g['passed'] for g in report['gates']),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        channel.close()
        connection.close()
        save(args.output, report)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()

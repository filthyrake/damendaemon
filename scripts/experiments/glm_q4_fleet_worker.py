"""#719 rung 5: one stage worker of the 4-host GLM-5.3 q4 fleet chain.

Stage cut (pre-declared on #719; every boundary a full-indexer layer, so
token selections never cross a host boundary):

  stage 1  labnode3  (A30,      CUDA) layers [0,34)   embeds + 3 dense + 31 MoE
  stage 2  rocky10   (A6000,    CUDA) layers [34,50) 16 MoE
  stage 3  3090box  (RTX 3090, CUDA) layers [50,66) 16 MoE
  stage 4  homer    (8060S,    ROCm) layers [66,78) 12 MoE + final norm + head

Per layer the worker runs the production glm_moe_dsa block composition:
fp8 attention (CudaCachedGlmAttention, shipped-fp8 projections INCLUDING the
indexer — selection exact, rung-4b decision #668), then the MoE FFN through
the rung-4a selected-expert streaming (host-resident canonical-q4_0 banks)
with the INLINE paired fp8 oracle: the same router selection replayed through
the shipped-fp8 codes of only the selected experts, on the ACTUAL chain
activations. Dense layers (0-2) keep their shipped-fp8 MLP unquantized.

Stages 2-4 listen (the previous stage connects out); TLS-authenticated
HostTensorChannel carries the [1, tokens, 6144] fp32 hidden state between
stages. Stage 1 requires the shared-CUDA guard env; stage 4 requires HIP and
runs the in-cell CoolingGuard startup check under the outer guard_homer
(rung-3b discipline, full homer stewardship outside).

Pre-declared gates (#719):
1. Per-MoE-layer paired-oracle parity on chain activations: rel RMS <= 0.15
   and max_abs <= 0.15 x absmax.
2. Finiteness of every block output; stage 4 additionally gates final-logits
   finiteness (lm_head stays shipped-fp32 — exact).
3. Routers/attention/indexer shipped-fp8/fp32 in both paths (structural).
4. Streaming bound: per-layer streamed bytes within (0, host bank bytes).
5. All guards clean; attempts retained.

No throughput/SLO/quality claim.
"""
import argparse
import hashlib
import json
import socket
from pathlib import Path

from scripts.experiments.split_gguf import save

STAGE_RANGES = {1: (0, 34), 2: (34, 50), 3: (50, 66), 4: (66, 78)}


def _hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--workdir', default='/tmp/damendaemon-g719r5')
    parser.add_argument('--banks-dir', default='/tmp/damendaemon-g719r5/banks')
    parser.add_argument('--pack-start', type=int)
    parser.add_argument('--pack-stop', type=int)
    # chain sub-cell range (guards cap at 1800s; boundaries MUST be full-indexer
    # layers so shared-indexer previous_indices never crosses a cell)
    parser.add_argument('--chain-start', type=int)
    parser.add_argument('--chain-stop', type=int)
    parser.add_argument('--hop-index', type=int, default=0,
                        help='HostTopology result index both ends of THIS hop agree on')
    parser.add_argument('--inbound-file', default=None,
                        help='resume: load the inbound hidden state from this file '
                             'instead of listening (crash-recovery path)')
    # inbound (stages 2-4)
    parser.add_argument('--address', default='0.0.0.0')
    parser.add_argument('--port', type=int)
    parser.add_argument('--ca')
    parser.add_argument('--cert')
    parser.add_argument('--key')
    parser.add_argument('--peer-cert', help='certificate of the previous stage')
    # outbound (stages 1-3)
    parser.add_argument('--next-address')
    parser.add_argument('--next-port', type=int)
    parser.add_argument('--next-cert', help='server certificate of the next cell')
    parser.add_argument('--next-server-name', default=None,
                        help='TLS server hostname of the next cell (defaults to next stage CN)')
    parser.add_argument('--timeout', type=float, default=7200.)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new report path required')
    start, end = STAGE_RANGES[args.stage]
    chain_start = args.chain_start if args.chain_start is not None else start
    chain_stop = args.chain_stop if args.chain_stop is not None else end
    if not (start <= chain_start < chain_stop <= end):
        raise ValueError('chain range must lie inside the stage range')
    packing = args.pack_start is not None or args.pack_stop is not None
    if not packing:
        # A cell opening on a shared-indexer layer would need previous_indices
        # from before its boundary; a cell closing before a shared layer would
        # have to hand the selection on. Both are forbidden — boundaries are
        # full-indexer (or dense) layers only.
        with open(f'{args.model}/config.json') as reader:
            schedule = json.load(reader)['indexer_types']
        for boundary in (chain_start, chain_stop):
            if start < boundary < end and schedule[boundary] != 'full':
                raise ValueError(f'chain boundary {boundary} is not a full-indexer layer')
    if not packing and (args.stage > 1 or chain_start > start) and not (args.port and args.ca and args.cert and args.key and args.peer_cert):
        raise ValueError('listening cells require --port and TLS arguments')
    if not packing and chain_stop == end and args.stage < 4 and not (args.next_address and args.next_port and args.next_cert):
        raise ValueError('cells ending a stage require next-stage connection arguments')
    if not packing and args.stage < 4 and not (args.ca and args.cert and args.key):
        raise ValueError('CUDA chain cells require TLS arguments')
    if packing and (args.pack_start is None or args.pack_stop is None):
        raise ValueError('pack mode needs both --pack-start and --pack-stop')

    import torch
    if args.stage == 4:
        from scripts.experiments.homer_cooling import CoolingGuard
        if not torch.version.hip or torch.cuda.device_count() != 1:
            raise ValueError('stage 4 requires exactly one HIP (ROCm) GPU')
        if torch.cuda.get_device_name().startswith('NVIDIA'):
            raise ValueError('stage 4 targets the AMD 8060S; refusing NVIDIA')
    else:
        from scripts.experiments.q5_cuda_oracle import require_guard
        require_guard()
        if torch.version.hip or torch.cuda.device_count() != 1:
            raise ValueError('stages 1-3 require exactly one NVIDIA GPU')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    cap_gib = 40 if args.stage == 4 else (26 if args.stage == 2 else 20)
    torch.cuda.set_per_process_memory_fraction(int(cap_gib * 2**30) / properties.total_memory)

    report = dict(passed=False, stage=args.stage, layers=[start, end],
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  tokens=args.tokens, gates=[], layer_gates=[])
    if args.stage == 4:
        CoolingGuard(report, limit_celsius=80).check(startup=True)
    save(args.output, report)

    listener = None
    connection = None
    try:
        import torch.nn.functional as F
        from damendaemon.host_endpoint import HostPeerListener, connect_peer
        from damendaemon.host_tensor_channel import HostTensorChannel
        from damendaemon.host_tls import certificate_pin, client_context, server_context
        from damendaemon.host_topology import HostTopology
        from scripts.experiments.glm_q4_fleet_common import (
            build_attention, load_attention_tensors, load_block_norms, load_dense_ffn_tensors,
            load_geometry, load_moe_tensors, moe_layer_forward, pack_q4_bank,
            resident_fp8_expert)

        device = torch.device('cuda:0')
        from damendaemon.glm_normalization import rms_norm
        from scripts.experiments.glm_q4_fleet_common import _load_tensors
        geometry = load_geometry(args.model)
        report['geometry'] = {k: geometry[k] for k in ('heads', 'nope', 'rope', 'value',
                                                      'top_k', 'dim', 'layers', 'vocab')}
        save(args.output, report)

        # ---- pack mode: guarded pre-cell that persists per-layer q4 banks.
        # The guards cap at 1800 s, so large stages pack in chunks; chain cells
        # load each layer's bank transiently (or pack in-flight when the file
        # is absent and --pack-stop was not given — the disk-poor host path).
        if args.pack_start is not None or args.pack_stop is not None:
            if args.pack_start is None or args.pack_stop is None or not (start <= args.pack_start < args.pack_stop <= end):
                raise ValueError('pack range must lie inside the stage range')
            import time
            banks_path = Path(args.banks_dir)
            banks_path.mkdir(parents=True, exist_ok=True)
            for layer in range(args.pack_start, args.pack_stop):
                if layer < geometry['first_dense']:
                    continue
                began = time.monotonic()
                tensors, experts, shards = load_moe_tensors(args.model, layer)
                host_bank, host_shared, router, bias, packed = pack_q4_bank(
                    tensors, experts, layer, device)
                del tensors
                torch.save(dict(layer=layer, experts=experts, host_bank=host_bank,
                                shared=host_shared, router=router, bias=bias,
                                packed_bytes=packed),
                           banks_path / f'layer{layer}.pt')
                del host_bank, host_shared, router, bias
                report['packed_layers'] = report.get('packed_layers', 0) + 1
                report['last_layer_seconds'] = round(time.monotonic() - began, 1)
                save(args.output, report)
                print(json.dumps(dict(event='packed', layer=layer,
                                      seconds=report['last_layer_seconds'])), flush=True)
            report['phase'] = 'pack_complete'
            save(args.output, report)
            print('FLEET_PACK_DONE', args.stage)
            return

        # Bind before the long phases so the upstream connect waits in backlog.
        if args.stage > 1 or chain_start > start:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw.bind((args.address, args.port))
            context = server_context(ca_file=args.ca, cert_file=args.cert, key_file=args.key)
            listener = HostPeerListener(raw, context, peer_pin=certificate_pin(args.peer_cert))
            print(json.dumps(dict(event='listening', stage=args.stage, port=args.port)), flush=True)

        # ---- chain banks: per-cell preload into host RAM (the disk read is
        # paid once per layer, not once per use).
        banks_path = Path(args.banks_dir)
        cell_banks = {}
        for layer in range(chain_start, chain_stop):
            if layer < geometry['first_dense']:
                continue
            bank_file = banks_path / f'layer{layer}.pt'
            if not bank_file.exists():
                raise ValueError(f'bank layer{layer}.pt missing — run the pack cells first')
            payload = torch.load(bank_file, map_location='cpu', weights_only=True)
            cell_banks[layer] = (payload['host_bank'], payload['shared'], payload['router'],
                                 payload['bias'], payload['packed_bytes'])
            del payload
        report['cell_packed_bytes'] = sum(entry[4] for entry in cell_banks.values())
        save(args.output, report)

        # ---- inbound chain state
        if args.inbound_file:
            hidden = torch.load(args.inbound_file, map_location=device, weights_only=True)
            if tuple(hidden.shape) != (1, args.tokens, geometry['dim']):
                raise ValueError('inbound resume file shape mismatch')
            report['received_hash'] = _hash(hidden)
            report['resumed_from_file'] = args.inbound_file
        elif args.stage > 1 or chain_start > start:
            connection = listener.accept(timeout=args.timeout)
            listener.close()
            listener = None
            config = json.load(open(f'{args.model}/config.json'))
            plan = HostTopology(config, [[0, 34], [34, 50], [50, 66], [66, 78]],
                                bytes(range(32)), quantization='fp8_reference',
                                context=128, context_limit=128, slots=2, chunk_size=8,
                                frame_bytes=64 * 1024**2,
                                mapped_vocab_size=config['vocab_size'], excluded_ids=())
            channel = HostTensorChannel(connection, plan.stage_identity(args.hop_index),
                                        1, max_frame_bytes=plan.frame_bytes, timeout=args.timeout)
            channel.handshake()
            hidden = channel.receive((1, args.tokens, geometry['dim']), torch.float32)
            channel.close()
            connection.close()
            connection = None
            report['received_hash'] = _hash(hidden)
        else:
            generator = torch.Generator().manual_seed(719)
            tokens = torch.randint(0, geometry['vocab'], (1, args.tokens),
                                    generator=generator).to(device)
            embed_names = ['model.embed_tokens.weight']
            from scripts.experiments.glm_q4_fleet_common import _load_tensors
            embed_tensors, _ = _load_tensors(args.model, embed_names)
            hidden = F.embedding(tokens, embed_tensors[embed_names[0]].float().to(device))
            del embed_tensors
            report['tokens_seed'] = 719

        # ---- chain phase
        selected = None
        for layer in range(chain_start, chain_stop):
            full = geometry['indexer_types'][layer] == 'full'
            attention_tensors, _ = load_attention_tensors(args.model, layer, full)
            attention = build_attention(attention_tensors, args.model, layer, device,
                                        geometry, full)
            (input_norm, post_norm), _ = load_block_norms(args.model, layer)
            input_norm, post_norm = input_norm.to(device), post_norm.to(device)
            normed = rms_norm(hidden, input_norm, geometry['epsilon'])
            if geometry['indexer_types'][layer] == 'full':
                attended, selected = attention.prefill(
                    normed, capacity=args.tokens, max_cache_bytes=2 * 1024**3)[:2]
            else:
                attended, _ = attention.prefill(
                    normed, capacity=args.tokens, max_cache_bytes=2 * 1024**3,
                    previous_indices=selected)[:2]
            del attention, attention_tensors, normed
            torch.cuda.empty_cache()
            residual = hidden + attended
            del attended
            ffn_input = rms_norm(residual, post_norm, geometry['epsilon'])
            if layer < geometry['first_dense']:
                dense_tensors, _ = load_dense_ffn_tensors(args.model, layer)
                expert = resident_fp8_expert(dense_tensors, f'model.layers.{layer}.mlp', device)
                y = expert.forward_batched(ffn_input.reshape(-1, geometry['dim'])).float()
                y = y.reshape(hidden.shape)
                gate = dict(layer=layer, kind='dense_fp8', streamed_bytes=0)
                del expert, dense_tensors
            else:
                host_bank, host_shared, router, bias, packed = cell_banks[layer]
                geometry['_bank_bytes'] = packed
                y, gate = moe_layer_forward(ffn_input, args.model, layer, host_bank,
                                             host_shared, router, bias, device, geometry)
                del host_bank, host_shared, router, bias
            gate['finite'] = bool(torch.isfinite(y).all())
            hidden = residual + y
            del residual, ffn_input, y
            report['layer_gates'].append(gate)
            save(args.output, report)
            print(json.dumps(dict(event='layer', layer=layer,
                                  parity=gate.get('parity'),
                                  rel=gate.get('rel_rms'))), flush=True)

        # ---- finish
        if chain_stop < end or args.stage < 4:
            config = json.load(open(f'{args.model}/config.json'))
            plan = HostTopology(config, [[0, 34], [34, 50], [50, 66], [66, 78]],
                                bytes(range(32)), quantization='fp8_reference',
                                context=128, context_limit=128, slots=2, chunk_size=8,
                                frame_bytes=64 * 1024**2,
                                mapped_vocab_size=config['vocab_size'], excluded_ids=())
            # Dump BEFORE connecting: a refused/failed connect must not lose
            # the computed state — the downstream resumes from this file.
            handoff_file = Path(args.workdir) / f'handoff-layer{chain_stop}.pt'
            torch.save(hidden.detach().cpu(), handoff_file)
            context = client_context(ca_file=args.ca, cert_file=args.cert, key_file=args.key)
            server_name = args.next_server_name or f'fleet-stage-{args.stage + 1}'
            connection = connect_peer(args.next_address, args.next_port, context,
                                      peer_pin=certificate_pin(args.next_cert),
                                      server_hostname=server_name,
                                      timeout=args.timeout)
            channel = HostTensorChannel(connection, plan.stage_identity(args.hop_index),
                                        0, max_frame_bytes=plan.frame_bytes, timeout=args.timeout)
            channel.handshake()
            report['sent_hash'] = _hash(hidden)
            channel.send(hidden.detach().cpu())
            channel.close()
            connection.close()
            connection = None
            report['gates'].append(dict(gate='handoff_complete', passed=True))
        else:
            from scripts.experiments.glm_q4_fleet_common import _load_tensors
            final_tensors, _ = _load_tensors(args.model, ['model.norm.weight', 'lm_head.weight'])
            from damendaemon.glm_normalization import rms_norm
            value = hidden[:, -1]
            value = rms_norm(value, final_tensors['model.norm.weight'].to(device),
                             geometry['epsilon'])
            logits = torch.nn.functional.linear(value, final_tensors['lm_head.weight'].to(device))
            finite = bool(torch.isfinite(logits).all())
            report['logits_absmax'] = float(logits.abs().max().item())
            report['gates'].append(dict(gate='final_logits_finite', passed=finite))
            del final_tensors, logits

        parity_gates = [g for g in report['layer_gates'] if 'parity' in g]
        report['gates'].append(dict(
            gate='per_layer_paired_oracle_parity',
            layers=len(parity_gates),
            worst_rel_rms=max(g['rel_rms'] for g in parity_gates),
            passed=all(g['parity'] for g in parity_gates)))
        report['gates'].append(dict(
            gate='streaming_bounded',
            passed=all(0 < g['streamed_bytes'] < g['host_bank_bytes']
                       for g in report['layer_gates'] if g.get('kind') != 'dense_fp8')))
        report['gates'].append(dict(gate='all_layers_finite',
                                    passed=all(g['finite'] for g in report['layer_gates'])))
        report['passed'] = all(g['passed'] for g in report['gates'])
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save(args.output, report)
        print('FLEET_STAGE_DONE', args.stage, 'passed' if report['passed'] else 'FAILED')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
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

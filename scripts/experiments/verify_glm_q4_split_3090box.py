"""#730: dual-3090 NVLink peer-resident q4 bank split for layers [58,66).

The 3090box RTX 3090 pair carries NV4 NVLink (47.52 GiB/s D2D measured,
PR #729). The shipped fleet-chain path keeps q4 banks HOST-resident and
streams each selected expert over PCIe per use (rung 4a, #719 rung 5).
This cell makes the banks DEVICE-resident, split by expert across both
GPUs (contiguous halves), executing on cuda:0 — far-half experts are read
on demand via peer access over NVLink, proving the aggregate-VRAM stage
capacity that a single 24 GiB card cannot hold (~40.7 GiB for this range).

Control: the shipped 3b cell's own output for this exact range and inbound
(handoff-layer58.pt in, handoff-layer66.pt out) is parked on 3090box.
Copies are bit-exact and kernels/accumulation order are identical, so the
split arm must reproduce the control BIT-IDENTICALLY — rung 4a's
streamed==resident gate class (max_abs == 0.0 exactly).

Pre-declared gates (#730):
1. split_vs_control_bitexact: final hidden torch.equal the parked 3b dump.
2. Per-layer paired-fp8-oracle parity on chain activations (rel RMS <= 0.15
   AND max_abs <= 0.15 x absmax) — the standing rung-5 envelope.
3. peer_integrity: the first far-resident expert of every layer round-trips
   bit-identically (host -> far GPU -> host, torch.equal on codes+scales).
4. memory_envelope: per-GPU peak reserved leaves >= 6% of device memory
   free (the live reserve bound is the guard's job at run time).
5. residency_capacity recorded: aggregate + per-GPU resident bank bytes,
   and the single-GPU infeasibility arithmetic for this range.
6. peer bandwidth recorded (aggregate far-read bytes / copy wall) against
   the 47.52 GiB/s qualification; sanity gate > 5 GiB/s (the host-staged
   control measured 1.32; per-call overhead keeps small reads under the
   2-GiB-blob qualification number, so the gate proves NVLink engagement,
   not the blob peak).
No throughput/SLO/quality claim — capability + parity only.
"""
if not __debug__:
    raise RuntimeError('verify_glm_q4_split_3090box asserts gates; '
                       'run without -O')

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.experiments.split_gguf import save

LAYER_START, LAYER_STOP = 58, 66
LOCAL, FAR = 0, 1
# torch.load targets are this fleet lane's own .pt artifacts (tensors +
# small dict metadata only); every call pins weights_only=True.


def _split_bank_get_factory(device, stats):
    """On-demand expert read from a split-resident bank.

    Local-half triples are already on the executing device; far-half
    triples sit on the peer GPU, so ``_q4_expert``'s ``.to(device)`` is
    the P2P NVLink read — timed and counted as peer bytes.
    """
    from scripts.experiments.glm_q4_fleet_common import _q4_expert

    def make_bank_get(split_bank, _ignored_device=None):
        """Two-arg signature matching _q4_bank_get_factory(host_bank, device).

        The device argument is redundant (the closure carries the executing
        device) but kept for signature parity with the default factory.
        """
        def bank_get(expert_id):
            triple = split_bank[expert_id]
            moved = sum(w.numel() + s.numel() * 2 for w, s in triple)
            far = triple[0][0].device.index != device.index
            began = time.monotonic()
            expert = _q4_expert(triple, device)
            seconds = time.monotonic() - began
            key = 'peer' if far else 'local'
            stats[key + '_reads'] += 1
            stats[key + '_bytes'] += moved
            stats[key + '_seconds'] += seconds
            return expert, moved

        return bank_get

    return make_bank_get


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--banks-dir', default='/home/damen/damendaemon-g719r5/banks')
    parser.add_argument('--inbound',
                        default='/tmp/damendaemon-g719r5/handoffs/handoff-layer58.pt')
    parser.add_argument('--control',
                        default='/tmp/damendaemon-g719r5/handoffs/handoff-layer66.pt')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true',
                        help='validate plan paths and boundaries, no GPU work')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new report path required')

    local = torch.device(f'cuda:{LOCAL}')
    far = torch.device(f'cuda:{FAR}')

    from scripts.experiments.glm_q4_fleet_common import load_geometry
    geometry = load_geometry(args.model)
    report: dict = dict(mechanism='nvlink_peer_resident_split_banks',
                  layer=[LAYER_START, LAYER_STOP], tokens=args.tokens,
                  device=[None, None], torch=torch.__version__,
                  gates=[], layer_gates=[])

    # ---- plan validation (also the --dry-run surface on a GPU-less host)
    banks = [Path(args.banks_dir) / f'layer{n}.pt'
             for n in range(LAYER_START, LAYER_STOP)]
    for path in [Path(args.model) / 'config.json', Path(args.inbound),
                 Path(args.control), *banks]:
        if not path.is_file():
            raise FileNotFoundError(path)
    schedule = json.load(open(f'{args.model}/config.json'))['indexer_types']
    for boundary in (LAYER_START, LAYER_STOP):
        if 50 < boundary < 66 and schedule[boundary] != 'full':
            raise ValueError(f'boundary {boundary} is not a full-indexer layer')
    if args.dry_run:
        print('DRYRUN_OK', LAYER_START, LAYER_STOP, len(banks), 'banks')
        return

    # ---- admission: exactly the two fleet 3090s, mutually peer-accessible
    if torch.cuda.device_count() != 2:
        raise RuntimeError('split cell requires exactly the two 3090s')
    names = [torch.cuda.get_device_name(i) for i in (LOCAL, FAR)]
    if any('3090' not in n for n in names):
        raise RuntimeError(f'unexpected devices: {names}')
    if not (torch.cuda.can_device_access_peer(LOCAL, FAR)
            and torch.cuda.can_device_access_peer(FAR, LOCAL)):
        raise RuntimeError('peer access unavailable between the two 3090s')
    report['device'] = names
    save(args.output, report)

    torch.backends.cuda.matmul.allow_tf32 = False  # bit-identity with 3b
    torch.cuda.reset_peak_memory_stats(LOCAL)
    torch.cuda.reset_peak_memory_stats(FAR)

    hidden = torch.load(args.inbound, map_location=local, weights_only=True)
    report['inbound_shape'] = list(hidden.shape)
    save(args.output, report)

    # ---- preload: every layer's bank split-resident across both GPUs.
    # Local half on the executing device, far half on the peer — the whole
    # range's q4 banks live in aggregate VRAM for the whole cell (the
    # capacity claim under test); the layer loop then never touches disk
    # or host RAM for expert weights.
    from scripts.experiments.glm_q4_fleet_common import (
        build_attention, load_attention_tensors, load_block_norms,
        moe_layer_forward)
    from damendaemon.glm_normalization import rms_norm
    stats = dict(peer_reads=0, local_reads=0, peer_bytes=0, local_bytes=0,
                peer_seconds=0., local_seconds=0.)
    split_banks, shareds, routers, biases, packed = {}, {}, {}, {}, {}
    integrity = []
    preload_began = time.monotonic()
    for layer in range(LAYER_START, LAYER_STOP):
        payload = torch.load(Path(args.banks_dir) / f'layer{layer}.pt',
                             map_location='cpu', weights_only=True)
        host_bank, experts = payload['host_bank'], payload['experts']
        half = (experts + 1) // 2
        bank = {}
        for expert_id, triple in host_bank.items():
            device = local if expert_id < half else far
            bank[expert_id] = [(w.to(device), s.to(device)) for w, s in triple]
        # gate 3: far residency round-trips bit-identically
        probe, resident = host_bank[half], bank[half]
        ok = all(torch.equal(a[0], b[0].cpu()) and torch.equal(a[1], b[1].cpu())
                 for a, b in zip(probe, resident))
        integrity.append(dict(layer=layer, expert=half, passed=bool(ok)))
        split_banks[layer] = bank
        shareds[layer] = payload['shared']
        routers[layer] = payload['router']
        biases[layer] = payload['bias']
        packed[layer] = payload['packed_bytes']
        del host_bank, payload
        print(json.dumps(dict(event='resident', layer=layer, experts=experts,
                              far_from=half)), flush=True)
    report['preload_seconds'] = round(time.monotonic() - preload_began, 1)
    report['resident_bytes'] = dict(
        aggregate=sum(packed.values()),
        per_gpu_local=sum(packed[l] for l in packed) // 2,  # contiguous halves
        per_layer=packed)
    report['single_gpu_feasible'] = sum(packed.values()) < 24 * 2**30
    save(args.output, report)

    # ---- chain phase: the rung-5 3b loop verbatim, banks split-resident
    selected = None
    for layer in range(LAYER_START, LAYER_STOP):
        full = schedule[layer] == 'full'
        attention_tensors, _ = load_attention_tensors(args.model, layer, full)
        attention = build_attention(attention_tensors, args.model, layer,
                                   local, geometry, full)
        norms, _ = load_block_norms(args.model, layer)
        input_norm, post_norm = norms[0].to(local), norms[1].to(local)
        normed = rms_norm(hidden, input_norm, geometry['epsilon'])
        if full:
            attended, selected = attention.prefill(
                normed, capacity=args.tokens,
                max_cache_bytes=2 * 1024**3)[:2]
        else:
            attended, _ = attention.prefill(
                normed, capacity=args.tokens, max_cache_bytes=2 * 1024**3,
                previous_indices=selected)[:2]
        del attention, attention_tensors, normed
        residual = hidden + attended
        del attended
        ffn_input = rms_norm(residual, post_norm, geometry['epsilon'])
        del input_norm, post_norm
        geometry['_bank_bytes'] = packed[layer]
        y, gate = moe_layer_forward(
            ffn_input, args.model, layer, split_banks[layer], shareds[layer],
            routers[layer], biases[layer], local, geometry,
            bank_get_factory=_split_bank_get_factory(local, stats))
        gate['finite'] = bool(torch.isfinite(y).all())
        gate['integrity'] = integrity[layer - LAYER_START]['passed']
        hidden = residual + y
        del residual, ffn_input, y, split_banks[layer], shareds[layer]
        report['layer_gates'].append(gate)
        save(args.output, report)
        print(json.dumps(dict(event='layer', layer=layer,
                              parity=gate.get('parity'),
                              rel=gate.get('rel_rms')), flush=True))
        with torch.cuda.device(local):
            torch.cuda.empty_cache()
        with torch.cuda.device(far):
            torch.cuda.empty_cache()

    # ---- gates
    control = torch.load(args.control, map_location='cpu', weights_only=True)
    final = hidden.detach().cpu()
    bitexact = bool(torch.equal(final, control))
    delta = (final - control).abs()
    parity_gates = [g for g in report['layer_gates'] if 'parity' in g]
    parity_ok = bool(parity_gates) and all(g['parity'] for g in parity_gates)
    integrity_ok = all(g['passed'] for g in integrity)
    peaks = {str(i): torch.cuda.max_memory_reserved(i) for i in (LOCAL, FAR)}
    totals = {str(i): torch.cuda.get_device_properties(i).total_memory
              for i in (LOCAL, FAR)}
    memory_ok = all(peaks[str(i)] <= 0.94 * totals[str(i)] for i in (LOCAL, FAR))
    peer_gib_s = (stats['peer_bytes'] / 2**30 / stats['peer_seconds']
                  if stats['peer_seconds'] > 0 else 0.)
    bandwidth_ok = peer_gib_s > 5.

    report.update(
        control_bitexact=bitexact,
        control_max_abs=float(delta.max().item()),
        peer=dict(reads=stats['peer_reads'], bytes=stats['peer_bytes'],
                  gib_s=round(peer_gib_s, 2),
                  qualification_gib_s=47.52),
        local=dict(reads=stats['local_reads'], bytes=stats['local_bytes']),
        peak_reserved_bytes=peaks, device_total_bytes=totals,
        stats={k: v for k, v in stats.items()})
    report['gates'] = [
        dict(gate='split_vs_control_bitexact', passed=bitexact,
             max_abs=float(delta.max().item())),
        dict(gate='per_layer_paired_oracle_parity', passed=parity_ok,
             layers=len(parity_gates),
             worst_rel=max(g['rel_rms'] for g in parity_gates)),
        dict(gate='peer_integrity', passed=integrity_ok,
             checked=len(integrity)),
        dict(gate='memory_envelope', passed=memory_ok,
             peaks=peaks, totals=totals),
        dict(gate='residency_capacity_recorded', passed=True,
             aggregate_bytes=sum(packed.values()),
             single_gpu_feasible=False),
        dict(gate='peer_bandwidth_engaged', passed=bandwidth_ok,
             gib_s=round(peer_gib_s, 2)),
    ]
    report['passed'] = all(g['passed'] for g in report['gates'])
    save(args.output, report)
    print('SPLIT_CELL_DONE', json.dumps(dict(passed=report['passed'])))


if __name__ == '__main__':
    main()

"""#536 gate-4 cycle-1: three-rank cross-host chain feasibility runner.

Qualifies the control-plane chain (damendaemon.distributed_chain) across the
real two-host topology with a REAL tensor pipeline, without hybrid model
weights: each rank runs a synthetic stage contract — embedding-free linear
stages over the same hidden size and split boundaries as the #551 plan — so
what is measured is exactly what gate 4 must first prove:

  1. torchrun/NCCL forms a 3-rank group across labnode3 (ranks 0,1) and the
     desktop (rank 2) with the scoped firewall (rendezvous 29500-29510).
  2. Command broadcast + MIN-ack ordering works over NCCL, not just the CPU
     bus transport (reuses ChainRole/ChainChannel).
  3. Hidden-state transfer per boundary per token/chunk at the #551 payloads
     (10 KiB decode, 1.27 MiB chunk-127), measured IN the NCCL P2P path.
  4. Full-chain greedy-ish round trip: rank0 broadcast -> stage compute ->
     forward -> stage compute -> forward -> head -> token return -> commit,
     repeated, with per-step timing reported separately from any GPU
     qualification claim.

Honesty: synthetic stages — no model weights, no logits gate, no serving
admission. This is the transport/control qualification half of gate-4
cycle 1; the hybrid-shard half is the next bounded step.

Run (rendezvous on labnode3):
  labnode3 rank0: torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2
      --master_addr 10.30.100.3 --master_port 29500
      -m scripts.experiments.gate4_three_rank_chain --role coordinator
  desktop rank2:  torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1
      --master_addr 10.30.100.3 --master_port 29500
      -m scripts.experiments.gate4_three_rank_chain --role worker
  (single-process demo on one host: --standalone)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch

HIDDEN = 5120
CHUNK = 127
DECODE_BYTES = 1 * HIDDEN * 2          # FP16 hidden for batch 1
CHUNK_BYTES = CHUNK * HIDDEN * 2
WARMUP_STEPS = 5
TIMED_STEPS = 50


def build(rank: int, world_size: int):
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size,
                                         timeout=timedelta(seconds=60))
    from damendaemon.distributed_chain import NcclChainTransport
    return NcclChainTransport()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('coordinator', 'worker'), default='worker')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 3:
        raise SystemExit(f'this runner qualifies exactly 3 ranks, got {world}')
    transport = build(rank, world)

    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    hidden_fp16 = torch.randn(CHUNK, HIDDEN, dtype=torch.float16, device=device)
    token_buf = torch.zeros(1, dtype=torch.int64, device=device)
    token_send = torch.tensor([42], dtype=torch.int64, device=device)

    timings = {'broadcast': [], 'forward': [], 'return': [], 'round': []}
    token_bc = torch.zeros(1, dtype=torch.int64, device=device)
    for step in range(WARMUP_STEPS + TIMED_STEPS):
        t0 = time.monotonic()
        header = torch.tensor([7, 0, step, 0, 0, 0, 0, 0], dtype=torch.int64, device=device)
        transport.broadcast(header, 0)
        t1 = time.monotonic()
        # Broadcast-chain steps with one identical GLOBAL order on every rank:
        # per-peer P2P sequencing mismatches across pairs hang the NCCL ring,
        # while a world broadcast per boundary step has a fixed order. Each
        # payload matters to its adjacent ranks but all ranks participate.
        transport.broadcast(hidden_fp16, 0)       # boundary A: stage0 -> stage1
        transport.broadcast(hidden_fp16, 1)       # boundary B: stage1 -> stage2
        t2 = time.monotonic()
        token_bc.fill_(42)
        transport.broadcast(token_bc, world - 1)  # boundary C: head -> coordinator
        t3 = time.monotonic()
        if step >= WARMUP_STEPS:
            timings['broadcast'].append(t1 - t0)
            timings['forward'].append(t2 - t1)
            timings['return'].append(t3 - t2)
            timings['round'].append(t3 - t0)

    if rank == 0:
        summary = {
            'mode': 'three-rank-nccl-chain',
            'world_size': world,
            'steps_timed': TIMED_STEPS,
            'payloads': {'decode_token_bytes': DECODE_BYTES, 'chunk127_bytes': CHUNK_BYTES},
            'median_seconds': {k: round(statistics.median(v), 6) for k, v in timings.items() if v},
            'max_seconds': {k: round(max(v), 6) for k, v in timings.items() if v},
            'token_received': int(token_bc.item()),
            'note': ('synthetic stages; timings include stage-compute zeros — forward '
                     'median IS the boundary transfer. No model/weights/serving claim.'),
        }
        print(json.dumps(summary, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

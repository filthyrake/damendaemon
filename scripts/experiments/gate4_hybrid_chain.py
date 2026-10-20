"""#536 gate-4 cycle-1b: hybrid shard execution over the three-rank NCCL chain.

Executes REAL hybrid stage shards of the qualified Qwen3.5-0.8B hybrid
geometry (CPU fixture numerics; no Q4 admission needed at this stage) across
the torchrun/NCCL rank chain, using the broadcast-chain pattern from
gate4_three_rank_chain (one world broadcast per boundary step, fixed global
order — the pattern that does not hang the ring).

Modes:
  reference : single-rank full-model executor on this device; writes the
              pinned greedy trace.
  chain     : torchrun 3-rank; each rank builds HybridLayerStage over its
              third of the layers from the SAME checkpoint, executes
              embed -> layers -> layers -> output_logits across the chain,
              compares greedy tokens with --compare <reference.json>.

Greedy equality is the gate. No serving claim, no performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch


SEED_TOKENS = list(range(16, 48))
GREEDY_TOKENS = 12


def build_stage(checkpoint: str, rank: int, world: int, device):
    from damendaemon.models import load as load_model
    model = load_model(checkpoint, 512, None)
    layers = model.config['num_hidden_layers']
    bounds = [0, round(layers / 3), round(2 * layers / 3), layers]
    start, end = bounds[rank], bounds[rank + 1]
    from damendaemon.hybrid_stage import HybridLayerStage
    stage = HybridLayerStage(model, start, end, max_batch=1, kernels='torch')
    # Move this shard's weights onto the rank's device.
    for key, tensor in list(stage.w.items()):
        stage.w[key] = tensor.to(device)
    for key, tensor in stage.kv.items():
        stage.kv[key] = tensor.to(device)
    for key, tensor in stage.conv.items():
        stage.conv[key] = tensor.to(device)
    for key, tensor in stage.state.items():
        stage.state[key] = tensor.to(device)
    if hasattr(stage, 'cos'):
        stage.cos = stage.cos.to(device)
        stage.sin = stage.sin.to(device)
        stage.arange = stage.arange.to(device)
    return stage, (start, end)


def run_reference(args):
    """Full-model single-rank HybridExecutor with EAGER TORCH kernels — the
    same arithmetic the chain stages use (kernels='torch'); fused Triton
    kernels would change FP16 reduction order and flip near-tie argmax."""
    os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(args.checkpoint, 512, None)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    executor.prefill(SEED_TOKENS)
    tokens = []
    token = SEED_TOKENS[-1]
    for _ in range(GREEDY_TOKENS):
        logits, next_ids = executor.decode(token)
        token = int(next_ids[0])
        tokens.append(token)
    report = {'mode': 'reference', 'checkpoint': args.checkpoint, 'seed': SEED_TOKENS,
              'greedy': tokens, 'kernels': 'torch',
              'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('reference', 'chain'))
    parser.add_argument('--checkpoint',
                        default='/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/'
                                'snapshots/2fc06364715b967f1860aea9cf38778875588b17')
    parser.add_argument('--output', default='/tmp/dd536-g4-reference.json')
    parser.add_argument('--compare', default=None)
    args = parser.parse_args()

    if args.action == 'reference':
        run_reference(args)
        return

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 3:
        raise SystemExit('gate-4 chain qualifies exactly 3 ranks')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank >= torch.cuda.device_count():
        local_rank = 0  # single-node 3-rank qualification reuses device 0
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=120))
    device = torch.device('cuda', local_rank)

    stage, bounds = build_stage(args.checkpoint, rank, world, device)
    tokens = []

    # Broadcast-chain pattern, fixed global order on all ranks:
    #   payload A (rank0's embedding output) -> boundary A
    #   payload B (rank1's layer output)     -> boundary B
    #   payload C (rank2 logits/token)       -> boundary C
    # To keep ONE tensor shape across all steps, every rank exchanges a
    # [1,1,HIDDEN] hidden buffer and rank2 separately broadcasts its token.
    # A CPU fixture geometry may not be 5120 wide; derive HIDDEN from config.
    hidden_width = stage.c['hidden_size']
    # FP32 residual: the executor's residual stream is FP32; rounding to FP16
    # at each boundary changes reduction inputs and flips near-tie argmax.
    hidden_buf = torch.zeros(1, 1, hidden_width, dtype=torch.float32, device=device)
    token_bc = torch.zeros(1, dtype=torch.int64, device=device)

    def chain_step(token_value, position):
        """One round: embed + layers0-7 (rank0) -> A -> layers8-15 (rank1)
        -> B -> layers16-23 + logits (rank2).

        Identical global broadcast order on every rank; the sampled token
        crosses the chain at boundary C. Every broadcast is completed with
        work.wait() before its payload is consumed (NCCL collectives run
        on an internal stream; the current stream would otherwise race
        the payload copy).
        """
        if rank == 0:
            hidden = stage.embedding(torch.tensor([[token_value]], device=device))
            hidden_buf.copy_(hidden.view(1, 1, hidden_width).to(torch.float32))
            # rank0 runs its own layers 0..7 BEFORE publishing to rank1.
            positions = torch.tensor([[position]], device=device)
            with torch.cuda.device(device):
                hidden = stage.forward_layers(hidden_buf.view(1, 1, hidden_width), positions,
                                              torch.tensor([0], device=device), False, 0)
            hidden_buf.copy_(hidden.view(1, 1, hidden_width).to(torch.float32))
        work = torch.distributed.broadcast(hidden_buf, 0)   # boundary A
        if work is not None:
            work.wait()
        if rank == 1:
            hidden = hidden_buf.view(1, 1, hidden_width)
            positions = torch.tensor([[position]], device=device)
            with torch.cuda.device(device):
                hidden = stage.forward_layers(hidden, positions,
                                              torch.tensor([0], device=device), False, 0)
            hidden_buf.copy_(hidden.view(1, 1, hidden_width).to(torch.float32))
        work = torch.distributed.broadcast(hidden_buf, 1)   # boundary B: rank1 -> rank2's input
        if work is not None:
            work.wait()
        if rank == 2:
            # The head stage runs AFTER receiving rank1's output; its own
            # hidden then feeds output_logits directly (only the sampled
            # token crosses the chain afterward).
            hidden = hidden_buf.view(1, 1, hidden_width)
            positions = torch.tensor([[position]], device=device)
            with torch.cuda.device(device):
                hidden = stage.forward_layers(hidden, positions,
                                              torch.tensor([0], device=device), False, 0)
            hidden_buf.copy_(hidden.view(1, 1, hidden_width).to(torch.float32))

    with torch.inference_mode():
        # Prefill the seed token-by-token (positions 0..31) so every rank's
        # attention history matches the reference's packed prefill.
        for offset, token_value in enumerate(SEED_TOKENS):
            chain_step(token_value, offset)
        # Decode: the head rank's argmax token feeds the next round.
        token = SEED_TOKENS[-1]
        for step in range(GREEDY_TOKENS):
            chain_step(token, len(SEED_TOKENS) + step)
            if rank == 2:
                logits = stage.output_logits(hidden_buf[:, -1])
                token = int(logits.argmax(-1)[0])
                token_bc.fill_(token)
            work = torch.distributed.broadcast(token_bc, 2)     # boundary C
            if work is not None:
                work.wait()
            if rank == 0:
                tokens.append(int(token_bc.item()))
                token = int(token_bc.item())

    if rank == 0:
        reference = json.loads(Path(args.compare).read_text()) if args.compare else None
        report = {'mode': 'chain', 'checkpoint': args.checkpoint,
                  'bounds': bounds, 'greedy': tokens,
                  'matches_reference': (reference['greedy'] == tokens) if reference else None}
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

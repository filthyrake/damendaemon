# #536 gate-4 cycle 1 — three-rank NCCL chain qualified (control + transport)

Date: 2026-10-19. Torchrun/NCCL across labnode3 (ranks 0,1: A30+A2) and
rocky10-desktop (rank 2: 3090). Runner:
`scripts/experiments/gate4_three_rank_chain.py`. Synthetic linear stages over
the #551 split boundaries — no model weights, no logits gate, no serving
claim. This is the transport/control half of gate 4; hybrid shards are next.

## Result: chain completes end to end

- torchrun `--nnodes=2 --rdzv_backend=static --master_port=29500` formed a
  3-rank NCCL group across both hosts (rendezvous on the scoped firewall
  port; NCCL negotiated its own ephemeral sockets through the source-scoped
  rich rules).
- 55 rounds of the full chain pattern: command broadcast (rank0→all) →
  hidden transfer A (0→1) → hidden transfer B (1→2) → token return C
  (2→0) → next round. `token_received: 42` on rank 0 = the head's payload
  crossed two host boundaries intact.
- NCCL 2.29.7 on BOTH hosts (labnode3's dllm-dev venv was upgraded from
  torch 2.6.0+cu124/NCCL 2.21.5 to torch 2.13.0+cu130 to match the desktop —
  the version mismatch was itself a finding: cross-version NCCL groups fail
  with opaque socket errors, not a clear diagnostic).

## Medians (50 timed steps after 5 warmup)

| step | median | max |
|---|---|---|
| command broadcast (8×int64) | 12.96 ms | 13.96 ms |
| boundary A+B (2×1.27 MiB chunk payload) | 0.187 ms | 0.307 ms |
| token return (8 B) | 0.144 ms | 0.269 ms |
| full round | 13.28 ms | 14.29 ms |

## Findings recorded for the hybrid-shard step

1. **NCCL version alignment is a hard prerequisite** across hosts; pin both
   venvs to the same torch/NCCL build (now 2.13.0+cu130/2.29.7).
2. **Firewall scoping**: labnode3 accepts the desktop on 29500-29510 +
   1024-65535 (NCCL's ephemeral socket listeners); the desktop accepts
   labnode3 likewise. Source-scoped to the two hosts, permanent+runtime.
3. **P2P sequencing constraint found the hard way**: per-peer
   `send`/`recv` calls issued in different global orders across pairs hang
   the ring (rank1's broadcast SeqNum=2 watchdog timeout). The runner's
   broadcast-chain pattern — one world broadcast per boundary step, fixed
   order on every rank — completes reliably. The real hybrid pipeline
   coordinator should either use this pattern or tag-matched P2P with a
   single global op sequence.
4. **The command broadcast (12.96 ms median) is unexpectedly the dominant
   cost** — larger than the two 1.27 MiB payload transfers combined
   (0.187 ms). Suspicion: the 8-word broadcast pays per-collective launch
   latency while the payload transfers ride warm channels; batching the
   command header INTO the payload transfer (or CUDA-graphing the round)
   is the obvious optimization. Recorded, not optimized — capacity work
   first.

## Not claimed

- No model execution, no greedy-vs-reference agreement, no serving
  lifecycle: the hybrid-shard half of gate-4 cycle 1 is the next bounded
  step. These timings are transport controls on idle GPUs, not performance
  results.

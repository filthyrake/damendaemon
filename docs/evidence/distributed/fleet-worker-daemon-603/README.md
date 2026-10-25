# Fleet worker daemon qualification — #603

Date: 2026-10-24. Runner: `scripts/experiments/fleet_worker_cell_603.py`
(worktree `issue-603-fleet-rpc-worker`). Records are the exact JSON each
role wrote; they carry measured durations but no wall-clock timestamps.

## What was qualified

One `WorkerDaemon` process on the rocky10-desktop RTX 3090 serving the
Qwen3.5-0.8B head stage (snapshot `2fc0636`, unquantized fp16, context
512, slots 2, layer split [0, 12) coordinator / [12, 24) daemon) over
mutual TLS with certificate pins and the `damendaemon-host/1` ALPN,
bound to `10.40.100.30:8300`. The session identity is the checkpoint's
content-derived fingerprint (`ecdd9261…a0312`), identical on all three
participating processes.

Three-way gate:

| Configuration | Rank 0 (coordinator) | Rank 1 (head) | Tokens |
|---|---|---|---|
| `pair.json` — direct sockets, no daemon | desktop 3090 (Triton) | desktop 3090 (Triton) | 3 prompts × 12 |
| `loopback.json` — through the daemon | desktop 3090 (Triton) | desktop 3090 (daemon, Triton) | 3 prompts × 12 |
| `remote.json` — through the daemon over the LAN | labnode3 A2 (`torch` kernels) | desktop 3090 (daemon, Triton) | 3 prompts × 12 |

`compare.json` verdict — **passed**:

- Greedy tokens identical across all three configurations (pair ==
  loopback == remote).
- The loopback session's FP32 boundary digests are bit-exact against the
  direct-socket pair (same devices, same protocol — the daemon adds no
  protocol drift).
- Identities agree across every process.
- The daemon served the two sequential sessions on one process
  (`daemon.json`: sessions = 2; one validation stage load plus one fresh
  stage per session, every stage closed after its session).
- The remote session ran the rank-0 stage on a different host and GPU
  than the rank-1 stage it drove.

Token-level agreement is the remote gate; bit-exact digests across
different rank-0 hardware are not claimed and were not expected.

## Commands (paths as run)

```
# desktop (RTX 3090, venv /home/damen/3090-damendaemon/venv)
credentials --tls-dir /tmp/fleet603/tls --server-ip 10.40.100.30
pair         --model <snapshot 2fc0636> --tls-dir /tmp/fleet603/tls --kernels triton
daemon       --model <snapshot 2fc0636> --tls-dir /tmp/fleet603/tls \
             --bind-host 10.40.100.30 --bind-port 8300 --sessions 2
coordinator  --connect-host 10.40.100.30 --connect-port 8300 \
             --server-hostname 10.40.100.30 --kernels triton   # loopback session
# labnode3 (A2, venv /home/damen/dllm-dev/venv, CUDA_VISIBLE_DEVICES=1)
coordinator  --model /tmp/fleet603/model --tls-dir /tmp/fleet603 \
             --connect-host 10.40.100.30 --connect-port 8300 \
             --server-hostname 10.40.100.30 --kernels torch      # remote session
# desktop
compare --pair pair.json --loopback loopback.json --remote remote.json \
        --daemon daemon.json
```

## Incident notes (what went wrong before the passing run)

- A first remote coordinator session completed against an earlier daemon
  instance but its record was lost: the operator had not created the
  output directory, so the role failed at `write()` after the session
  ended (and that daemon then correctly exited after its configured
  `--sessions 2`). The passing run above was taken end-to-end on one
  fresh daemon instance.
- The remote coordinator was originally planned on the Mac (MPS). The
  Mac venv (torch 2.6.0) predates the public `torch.mps.compile_shader`
  that the real-model Metal kernels (`hybrid_metal`, `quantized_metal`)
  require, so any real-model device stage fails there today; synthetic
  fixtures avoid those kernels, which is why the MPS unit tests pass.
  The remote rank was moved to labnode3's A2 and the cell runs
  unquantized fp16 for that reason; quantized device cells are #600
  evidence.
- A client presented with TLS material from a different CA is refused
  with a certificate-verification failure, as the mutual-TLS design
  intends; the qualifying client used the daemon host's CA.

## Scope limits

Bounded qualification of the daemon lifecycle and protocol fidelity:
exclusive-session admission, per-session stage freshness, sequential
session reuse, refusal/failure containment, and close semantics are
covered by `tests/test_worker_daemon.py` over real TLS loopback. Not
qualified: concurrent sessions, model switching on a live daemon, HTTP
serving admission, throughput/latency, and automatic restart. HOMER was
offline (thermal pad upgrade) and took no part.

## #603 follow-ups (2026-10-25): session queue, live model switching, deployment

Three recorded follow-ups landed as CPU-loopback work in
`tests/test_worker_daemon.py` (no new remote cell; the LAN cell above
remains the only remote qualification):

- **Bounded session queue** (`session_queue=1..16`): FIFO parking of
  arrivals while the session slot is held; the newest arrival is refused
  when the bound is hit; a waiter whose slot never frees within the
  daemon timeout is dropped and the daemon keeps serving.
- **Live model switching** (`additional_models` on both ranks, at most
  two entries, identical decoder configs, compatible unquantized
  host-resident layouts, sources detached from the live stage): the
  table is pinned at handshake, `switch_model` runs on idle sessions,
  both models round-trip with distinct greedy outputs, a fresh session
  reopens on the primary, and a table-size mismatch refuses the session
  on both sides instead of deadlocking.
- **Deployment note** (systemd unit pattern + out-of-band mutual-TLS
  provisioning + loopback default): the "Deployment note (gate 5)"
  section of `docs/multi-gpu.md` closes the last never-started item of
  the original #603 acceptance list.

Not qualified: quantized or device-boundary switch tables (rejected with
an explicit reason at construction), switch latency under load, and
concurrent sessions (still one exclusive slot).

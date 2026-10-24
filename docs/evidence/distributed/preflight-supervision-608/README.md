# #608 delivery: preflight, transport-version agreement, supervised rejoin (2026-10-24)

## Delivered (acceptance items 1-3, CPU-bus verified; live two-rank cell inventory-gated)

1. **`damendaemon preflight`** (bounded, read-only, host-local): torch build +
   version, NCCL/RCCL identity, GPU inventory with free memory, optional peer
   TCP probe of the rendezvous port, one pass/fail per item, gate-4 findings
   encoded as actionable text. Exit 0 only when every item passes.
2. **Cross-rank transport-version agreement**: the StagePair startup digest now
   carries the transport identity (torch version/build, NCCL or RCCL version,
   runtime) alongside the model table and session bounds; a mismatch raises an
   explicit error naming both ranks' values and the fix ("align venvs before
   rendezvous", gate 4) instead of an opaque socket failure at NCCL init.
3. **Supervised mode** (`distributed-serve --supervise`): dead-peer
   classification from the existing heartbeat/frame-deadline poisoning
   (timeout/NCCL/connection → 'peer-dead'), honest failed `/health` while the
   listener stays up, and a bounded operator-triggered rejoin
   (`POST /v1/distributed/rejoin`, 202/409) that spawns a fresh rank-1 worker
   and pair per generation and resumes serving with empty slots. Generation
   bound (default 3) then terminal — no automatic restart, no in-flight
   request replay, both disclosed. Implemented without touching the qualified
   non-supervised path.

## Evidence captured live (desktop 3090)

`preflight-desktop-3090.json`: real run on the desktop — torch 2.13.0+cu130 /
NCCL 2.29.7, RTX 3090 with UUID and 23.34 GiB free, peer probe to
10.30.100.3:29500 correctly classified as connection-refused with the
actionable note; exit code 1 with a failing item and 0 on the all-pass run
(verified both ways). This is the host side of the gate-4 alignment workflow.

## Honest boundaries

- The CPU-bus failure injection (rank death, NCCL abort, build failures,
  generation exhaustion) is covered by 14 unit tests in
  `tests/test_distributed_supervision.py` (CI module added). The live
  two-rank supervised cell and the cross-host chain are **inventory-gated**:
  the desktop has a single CUDA device (the B580 is Intel), HOMER is a single
  ROCm APU, and labnode3's A30/A2 are user-reserved — same gating as #602.
  The all_gather agreement path and the supervisor's real subprocess respawn
  remain unverified live until that cell can run.
- No claims of automatic restart, request replay, or graceful HTTP errors for
  process death mid-collective beyond the existing disclosures.

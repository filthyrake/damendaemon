# Cross-host probe run — 2026-10-24 incident record (WIP, run not completed)

Status: **paused mid-investigation** (user pause request). Branch is WIP;
nothing here is a qualification claim.

## What ran

`probe_distributed_gpu.py` (this branch, with transcript capture), socket
transport forced (`NCCL_NET=Socket`), rank 0 = labnode3 A30
(`CUDA_VISIBLE_DEVICES=0`), rank 1 = rocky10-desktop 3090
(UUID-pinned), rendezvous `tcp://10.30.100.3:29500`, `NCCL_DEBUG=INFO`,
`NCCL_DEBUG_SUBSYS=INIT,NET`. Two earlier launch attempts timed out in
the TCPStore join (31 s) before NCCL started — orchestration now polls
for the listener before firing rank 1. A same-host A30+A2 control run of
the same probe completed its full 16-operation grid cleanly.

## Findings so far

1. **Cross-host `all_reduce` (4 KiB) completed with real measurements**:
   rank1.json holds 20 timed iterations, median 1.36e-4 s
   (`/tmp/rdma602/rank1.crosshost-partial.json` on labnode3, preserved).
   NCCL socket transport, store join, and both GPUs executed correctly
   cross-host.
2. **Deadlock at the first cross-host `round_trip`**: both ranks
   completed the 4 KiB all_reduce, entered the 4 KiB P2P round-trip, and
   spun at ~100% CPU for 77+ minutes (rank 0 process state R, last file
   write 04:47:53, killed 06:05). `work.wait()` on the P2P ops has no
   deadline and the 30 s process-group timeout did not fire — this is the
   known "no outer process deadline" gap docs/multi-gpu.md already
   records, now reproduced as a hard hang on the cross-host P2P path.
   Same-host P2P round-trips completed fine. NEXT: reproduce with NCCL
   debug transcript captured (blocked by finding 3), inspect whether the
   desktop-side firewall drops one P2P direction (rank1→rank0 listener
   vs rank0→rank1), and add an outer deadline to the probe's run paths
   before any rerun.
3. **Transcript tee bug (real, unfixed)**: on BOTH labnode3 runs (the
   clean local pair and the deadlocked cross-host pair) the transcript
   file is 0 bytes while the console passthrough worked — `file.write`
   never landed though `os.write(saved, …)` did. The macOS unit test
   passes, so this is Linux-specific (fork/NCCL-proxy interaction with
   the pump thread is the prime suspect). NEXT: reproduce minimally on
   labnode3 (fork + fd-2 writes), fix `StderrTranscript`, re-run tests.

## Cleanup state

Both rank processes killed (labnode3 pid 806586 SIGTERM; desktop pid
612554 SIGKILL after SIGTERM failed); GPUs idle. Staging left in place
for resume: `/tmp/rdma602/` on labnode3 (code, rank0/rank1 jsons +
console logs, crosshost-partial copy) and `/tmp/rdma602/` on the desktop
(code, rank1.json). Rendezvous port 29500 free.

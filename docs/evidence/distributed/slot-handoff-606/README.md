# KV/prefix handoff across ranks (#606) — design note, protocol, CPU + live cells

Issue #606. Branch `issue-606-kv-handoff`. CPU implementation cell plus the
live labnode3 A30→A30+A2 correctness cell (gate 3), both below.

## Gate 1 — design note (arithmetic before building)

Real fleet fixtures, real configs from the labnode3 HF cache
(`Qwen3.8-27B` snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
`Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`),
state geometry from `hybrid_layout.hybrid_state_bytes` (the same formula the
capacity planner uses):

- **Qwen3.8-27B**: 64 layers (16 full-attention, 48 linear-attention), KV
  heads 4, head_dim 256 → attention KV 32,768 elements/token = 0.125 MiB
  FP32 / 0.0625 MiB FP16 / 0.035 MiB Q8_0. Linear-attention fixed state
  (recurrent + conv, FP32): 12.75 MiB/layer × 48 = 598 MiB — independent of
  prefix length.
- **Qwen3.5-0.8B**: 24 layers (6 full, 18 linear), KV heads 2, head_dim 256
  → attention KV 6,144 elements/token = 0.0234 MiB FP32. Linear fixed state:
  4.29 MiB/layer × 18 = 75.8 MiB.

One-slot handoff totals (attention KV + fixed linear state) and wire time at
the two measured bands this fleet has on record — in-process A30→A2 device
copies 5.26–5.80 GiB/s at ≥8 MiB (#604 evidence), and the ~1.25 GB/s 10GbE
line-rate class for the cross-host variant:

| Prefix | 27B FP32 | 27B Q8_0 | 0.8B FP32 | wire @5.3 GiB/s (27B FP32) | wire @1.25 GB/s (27B FP32) |
|---|---|---|---|---|---|
| 512 | 0.632 GiB | 0.587 GiB | 0.084 GiB | 0.12 s | 0.54 s |
| 2048 | 0.820 GiB | 0.640 GiB | 0.119 GiB | 0.15 s | 0.70 s |
| 32768 | 4.570 GiB | 1.695 GiB | 0.822 GiB | 0.86 s | 3.93 s |

(The 32K FP32 attention-KV figure, 4.000 GiB, reproduces the number quoted
in the issue from the #535/#70 27B cell.)

Break-even — handoff beats re-prefill on the same link iff the receiver's
re-prefill rate is below `tokens / wire_seconds`:

| Prefix | 27B FP32 @1.25 GB/s | 27B FP32 @5.3 GiB/s | 0.8B FP32 @1.25 GB/s |
|---|---|---|---|
| 512 | 948 tok/s | 4.2K tok/s | 7.0K tok/s |
| 2048 | 2.9K tok/s | 12.9K tok/s | 20.5K tok/s |
| 32768 | 8.3K tok/s | 37.0K tok/s | 46K tok/s |

Decision reading (honest): this repo has **no measured 27B 32K prefill
rate** to compare against — the break-even table is the decision envelope,
not a verdict. What the arithmetic does dissolve: (1) the fixed
linear-attention state dominates short-prefix handoffs (0.57 GiB of the
27B's 0.632 GiB at 512 tokens is length-independent, so the per-request
handoff cost does not shrink with short prompts); (2) at 32K the FP32
attention KV dominates and Q8_0 KV quarters it, exactly the regime the
issue's disaggregation case named; (3) the bounded cell the numbers justify
is the **in-process A30→A2 pair with the 0.8B** — 0.084 GiB at 512 tokens
(~15 ms at the #604 measured device-copy band), i.e. a correctness cell
whose handoff cost is negligible, not a throughput cell. Throughput claims
are out of scope for this issue by its own gate 5.

## Gate 2 — protocol extension (landed)

`stage_protocol.py` VERSION 5 → 6; new `KV_HANDOFF` operation. A handoff
INSTALLS a populated slot (rows tokens of transferred KV/conv/recurrent
state) as a fresh generation: receiver slot must be empty, claim the next
generation, and carry `position == rows` — a handoff is never partial, the
payload carries the whole populated prefix. Stale generations, occupied
slots, multi-slot handoffs and context overflow refuse exactly like
`PREFILL_CHUNK`'s stale-offset rejection. The receiver's cursor commits only
in the commit path — after the acknowledged round. VERSION bump prevents
v5 peers from silently accepting the new semantics.

`host_stage_channel.py`: `finish` accepts `KV_HANDOFF`; the handoff round is
control-only on the wire — payload tensors travel over the tensor transport
between the channel's begin and finish, exactly like execution activations.

## Payload module (landed)

`slot_handoff.py` — `export_slot` / `import_slot` / `release_slot`:

- Export clones one populated slot's attention KV (plain or Q8_0-packed,
  scales included), conv and recurrent state to bounded CPU tensors with a
  content-derived identity inventory (config, quantization, KV mode, compute
  dtype, context, batch, layer range, position, per-tensor dtype/shape/bytes).
- Import installs into a fresh slot of a compatible stage; every agreement
  check is content-derived; a superset payload feeds each rank of a
  layer-split pair its own slice (entries outside the rank's layer range are
  skipped); the stage cursor commits only after every tensor is installed;
  a refused handoff raises before touching receiver state.
- Release zeroes the sender's slot after the acknowledged handoff (mirrors
  worker RELEASE).
- `HybridPipeline.install_handoff(payload, slot, generation)` runs the
  in-process pair path: every stage imports its slice, then the pipeline
  cursor commits; per-stage handoff telemetry is exposed in `status()`.

## Gate 4 — telemetry (landed)

`status()['handoffs']` carries per-stage `import_bytes`, `tensors`,
`import_seconds`, `slot`, `generation`, `position`; `export_slot` returns
`export_bytes`/`export_seconds`; `release_slot` reports the released
position.

## CPU correctness cell (this PR)

`tests/test_slot_handoff.py` on the 4-layer CPU hybrid fixture
(2 linear-attention + 2 full-attention layers, KV on layers 1 and 3):

- A prefilled sender's slot exported mid-flight and installed into a fresh
  two-stage pipeline decodes **greedy-equal** to (a) the sender's own
  single-executor continuation and (b) a control pipeline recomputing the
  same prompt — token-for-token identical chains, and the per-layer state
  byte-equal (`rtol=0, atol=0`) against the handoff source. The receiver
  never re-runs the prompt; the sender's first sampled token travels with
  the handoff (the disaggregated-prefill contract).
- Refusals: occupied target slot, tampered config/context/dtype/position,
  receiver layer range not covered by the payload, payload missing an
  inventory tensor, empty inventory — all before receiver state is touched.
- Protocol state machine (test_stage_protocol.py): handoff installs a fresh
  generation that then decodes and chunk-continues like a prefilled slot;
  stale/next-generation claims, partial handoffs (position != rows),
  occupied slots, multi-entry handoffs, context overflow and the v5 wire
  version all refuse.
- Wire round (test_host_stage_channel.py): a KV_HANDOFF round moves the
  payload over the tensor channel between begin and finish, commits the
  cursor only after the acknowledged finish, and the installed slot decodes
  over the same wire.

58 tests green across `test_stage_protocol`, `test_host_stage_channel`,
`test_slot_handoff`, `test_worker_daemon`, `test_hybrid_stage`,
`test_host_tensor_channel` (Mac venv, torch 2.6 CPU).

## Gate 3 — live cell (labnode3 A30 → A30+A2 in-process pair)

Runner `handoff-cell-606.py` (this directory), executed under
`scripts.experiments.guard_shared_cuda` with the ordered A30+A2 UUID pair;
result in `handoff-cell.json`, guard report in `guard.json`, log in
`cell.log`. Qwen3.5-0.8B unquantized (revision `2fc0636…`), context 512,
max_batch 2, kernels `torch`, use_graph off; split pipeline stage 0 =
layers [0,12) on the A30 (`cuda:0`), stage 1 = [12,24) on the A2
(`cuda:1`); sender = single-device HybridExecutor on the A30; control =
a second split pipeline recomputing the same prompt.

- Pinned 64-token prompt; the sender's slot exported at the prompt-only
  state BEFORE its own continuation advanced the live slot (export clones).
- **Greedy-equal, token-for-token (12/12)**: the handoff pipeline's
  continuation (decoding from the sender's first sampled token, `22287`)
  equals the sender's own A30 continuation AND the recomputed control
  pipeline. `sender_continuation_equals_handoff: true`,
  `recomputed_control_equals_handoff: true`, `completed: true`, guard
  `cleanup_complete: true`; both GPUs back to 0 MiB after the run.
- Telemetry (real, recorded in the result JSON): export 20,545,536 bytes
  over 42 tensors in 0.052 s; per-stage install 10,272,768 bytes over 21
  tensors (half the layers each) in 0.0045 s / 0.0121 s; sender prefill
  1.08 s; sender slot released after the acknowledged handoff (live
  position 75 = 64 prompt + 11 decodes at release time).
- Measured-vs-design-note delta, disclosed: the live export measured
  20.5 MiB at 64 tokens (fp16 compute path), about half the fp16-scaled
  capacity-formula estimate for the 0.8B fixture — the design note's
  fixture table is conservative for the 0.8B; the 27B 32K FP32 figure
  (4.000 GiB) reproduces the recorded #535/#70 cell and is the regime the
  arithmetic was for.
- Correctness cell only — no throughput claim; the total 11.6 s wall time
  includes two model loads and four stage constructions, not a handoff
  latency measurement.

## Honest boundary

- Worker (`HostHybridWorker`) transport integration — a KV_HANDOFF arriving
  at a host worker — is not wired in this PR; the command would fail loudly
  at the execution branch. Cross-host handoff over the authenticated channel
  remains future scope.
- No throughput claim is made or implied by the wire-time columns above;
  they are arithmetic from recorded bands (#604 device copies, 10GbE line
  rate), not measurements of this implementation.

## Disaggregated-prefill contract (recorded for the live cell)

The handoff payload is pure state. The receiver cannot produce the first
continuation token from state alone (the prompt's last-token logits stay on
the sender), so the sender's first sampled token travels with the handoff;
the receiver's decode starts from it at the installed position. This mirrors
standard prefill/decode disaggregation and is what the live cell must assert.

"""#601: Qwen3.8-27B mixed per-rank precision over the three-device chain.

The gate-4-qualified topology (labnode3 A30 rank 0 + A2 rank 1, desktop
RTX 3090 rank 2 + output head) with PER-RANK weight formats (#601). Two
passes over the pinned gate-4 trace (SEED = range(16,48), context 2048,
batch 1, 12 greedy tokens, standard autoregressive decode protocol):

  pass A (reference): GATE4_FORMATS=q4_0,q4_0,q4_0 — the qualified uniform
      chain, same split, standard protocol (the recorded gate-4 greedy used
      the older prompt[-1] convention, so the reference is re-run here).
  pass B (mixed):     GATE4_FORMATS=q4_0,q4_0,none — ranks 0/1 keep the
      compensated Q4 chain admission; rank 2 runs the unquantized FP16
      Triton path. Both passes use the SAME layer split [0, 24, 44, 64] so
      the two inter-rank boundaries carry identical upstream activations:
      rank 0 (Q4) and rank 1 (Q4) execute the identical qualified path in
      both passes, and any divergence is isolated inside rank 2.

The mixed split the per-rank-format planner would choose for the same
budgets is reported alongside (capacity demonstration); the run itself
pins the reference split so the comparison stays exact.

Claim class per house rules: boundary transport is compared bit-exactly
(SHA-256 per round); the mixed head's logits are reported with max_abs and
rms_rel exactly as measured, and greedy token agreement is recorded as it
lands — mixed precision across ranks does not compose bit-exactly by
construction.

Run under torchrun --nnodes=2 --node_rank={0|1} --nproc_per_node={2|1}
--rdzv_backend=static --master_addr=10.30.100.3 --master_port=29500.
"""
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

SNAPSHOT_LABNODE = '/tmp/q3827b'
SNAPSHOT_DESKTOP = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
SEED = list(range(16, 48))
GREEDY = 12
REFERENCE_SPLIT = [0, 24, 44, 64]
BUDGETS_MIB = [22528, 13312, 22528]
REPORT_REFERENCE = Path('/tmp/dd601-reference.json')
REPORT_MIXED = Path('/tmp/dd601-mixed.json')
CAPTURES = Path('/tmp/dd601-captures')
FORMATS = os.environ.get('GATE4_FORMATS', 'q4_0,q4_0,none').split(',')


def admit_quantized_context():
    """Bounded chain-probe admission, the gate-4 probe convention: context
    admission (512 -> 512..32768) and the A30-primary pair anchor (the
    remote tail rank) are substituted, recorded as part of what this gate
    qualifies. Opt-in, accelerator, architecture, precision and vision
    guards stay intact; production admission is unchanged."""
    import damendaemon.hybrid_q4_policy as hybrid_q4_policy
    import damendaemon.hybrid_precision as hybrid_precision
    from damendaemon.device import selected_accelerator
    from damendaemon.vision_policy import enabled as vision_enabled
    original_q4 = hybrid_q4_policy.validate
    original_select = hybrid_precision.select

    def validate(config, max_context=None):
        if max_context is not None and not 512 <= max_context <= 32768:
            raise ValueError('Q4 chain probe context is bounded to 512..32768')
        accelerator = selected_accelerator()
        if (os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4', '0') != '1'
                or accelerator not in ('rocm', 'cuda')
                or config.get('model_type') != 'qwen3_5_text'
                or os.environ.get('DAMENDAEMON_HYBRID_PRECISION') != 'compensated'
                or vision_enabled()):
            original_q4(config, 512)
            raise ValueError('mixed-precision chain probe keeps the Q4 opt-in, '
                             'accelerator, architecture, precision and vision guards')
        return None

    def select(model, use_graph, max_batch, kernels):
        from copy import copy as clone
        if model.quantization == 'q4_0':
            validate(model.config, model.max_context)
            admission = clone(model)
            admission.max_context = 512
            return original_select(admission, use_graph, max_batch, kernels)
        # Unquantized ranks take the standard qualified FP16 path unchanged.
        return original_select(model, use_graph, max_batch, kernels)

    hybrid_q4_policy.validate = validate
    hybrid_precision.select = select


def per_rank_environment(rank):
    """Each rank declares its OWN format's opt-ins (the #601 CLI shape)."""
    fmt = FORMATS[rank]
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
    os.environ['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = (
        '22528' if rank == 2 else '22528,13312')
    os.environ['DAMENDAEMON_HYBRID_PREFILL'] = 'scan'
    if fmt == 'q4_0':
        os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
        os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
        os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN'] = '1'
        os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    else:
        os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'float16'


def build_stage(rank, world, device):
    from damendaemon.models import load as load_model
    from damendaemon.hybrid_stage import HybridLayerStage
    checkpoint = SNAPSHOT_LABNODE if rank in (0, 1) else SNAPSHOT_DESKTOP
    fmt = FORMATS[rank]
    quantization = None if fmt == 'none' else fmt
    model = load_model(checkpoint, 2048, quantization)
    layers = model.config['num_hidden_layers']
    # The planner's own mixed split for these budgets/formats (reported as
    # the capacity demonstration), plus a feasibility check that the pinned
    # reference split fits every rank in its own format.
    from damendaemon.hybrid_layout import (choose_stage_splits_formats,
                                            stage_budget_formats,
                                            stage_pricing_table)
    pricing = stage_pricing_table(model.weights, getattr(model, 'scales', None),
                                  model.config)
    planned = choose_stage_splits_formats(pricing, model.config, model.max_context,
                                          1, 2, [mib * 1024**2 for mib in BUDGETS_MIB],
                                          FORMATS)
    if not planned['storage_plan_found']:
        raise SystemExit('mixed planner found no plan: ' + str(planned['reason']))
    bounds = list(REFERENCE_SPLIT)
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        budget = stage_budget_formats(pricing, model.config, model.max_context, 1, 2,
                                      start, end, FORMATS[index])
        if budget['required_memory_bytes'] > BUDGETS_MIB[index] * 1024**2:
            raise SystemExit(f'reference split does not fit rank {index} '
                             f'({FORMATS[index]}): {budget}')
    stage = HybridLayerStage(model, bounds[rank], bounds[rank + 1],
                             max_batch=1, kernels='triton')
    for group in ('w', 'kv', 'conv', 'state'):
        for key, tensor in list(getattr(stage, group).items()):
            getattr(stage, group)[key] = tensor.to(device)
    if hasattr(stage, 'cos'):
        stage.cos = stage.cos.to(device)
        stage.sin = stage.sin.to(device)
        stage.arange = stage.arange.to(device)
    planned_split = [0] + list(planned['splits']) + [layers]
    return model, stage, bounds, planned_split, layers, checkpoint


class BoundaryDigest:
    """SHA-256 of every FP32 boundary activation; head logits kept for deltas."""

    def __init__(self, rank):
        self.rank = rank
        self.boundaries = []   # (round, shape, digest)
        self.logits = []       # (round, digest, argmax)
        self.logit_tensors = []

    def hidden(self, tensor, label):
        data = tensor.detach().to(torch.float32).contiguous().cpu()
        flat = data.view(-1).view(torch.uint8).tolist()
        digest = hashlib.sha256(bytes(flat)).hexdigest()
        self.boundaries.append(dict(round=label, shape=list(data.shape),
                                    sha256=digest))

    def head(self, logits, label):
        data = logits.detach().to(torch.float32).contiguous().cpu()
        flat = data.view(-1).view(torch.uint8).tolist()
        digest = hashlib.sha256(bytes(flat)).hexdigest()
        self.logits.append(dict(round=label, sha256=digest,
                                argmax=int(data.argmax(-1)),
                                max_logit=float(data.max())))
        self.logit_tensors.append(data)


def capture_hooks(stage, worker, capture, sink=None, bounds=None):
    """Digest boundary activations (ranks 1/2) and head logits (rank 2).

    run_chain_round broadcasts every rank's payload to every rank; a rank's
    `received` argument is the previous stage's FP32 boundary output, and
    the head rank's stage.output_logits sees the final logits before the
    protocol argmax. Hooks stay passive: they never alter payloads. When a
    sink path is given, every captured round is ALSO flushed to disk
    immediately — a hung teardown (observed once: NCCL teardown stalled
    after STOP with the capture writeback still pending) must not lose the
    qualification data.
    """
    original_forward = worker._stage_forward
    counter = [0]

    def flush():
        if sink is None:
            return
        sink.parent.mkdir(parents=True, exist_ok=True)
        if capture.boundaries:
            sink.with_suffix('.json').write_text(json.dumps(
                dict(rank=worker.rank, formats=FORMATS, bounds=bounds,
                     boundary_digests=capture.boundaries), indent=2))
        if capture.logit_tensors:
            torch.save(capture.logit_tensors,
                       sink.with_name(sink.name + '-logits.pt'))

    def forward(command, tokens=None, received=None):
        label = counter[0]
        counter[0] += 1
        if received is not None:
            capture.hidden(received, label)
        result = original_forward(command, tokens, received)
        flush()
        return result

    worker._stage_forward = forward
    if worker.head:
        original_logits = stage.output_logits

        def output_logits(hidden):
            logits = original_logits(hidden)
            capture.head(logits, counter[0] - 1)
            return logits

        stage.output_logits = output_logits
    return capture


def main():
    admit_quantized_context()
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 3 or len(FORMATS) != 3:
        raise SystemExit('the mixed-precision chain qualifies exactly 3 ranks '
                         'and 3 per-rank formats')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
    per_rank_environment(rank)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=1800))
    device = torch.device('cuda', local_rank)

    model, stage, bounds, planned_split, layers, checkpoint = build_stage(rank, world, device)

    from damendaemon.stage_protocol import StageState
    from damendaemon.distributed_chain import (ChainChannel, ChainRole,
                                               NcclChainTransport)
    from damendaemon.stage_identity import chain_contract

    state = StageState(epoch=1, context=stage.context, slots=stage.max_batch,
                       context_limit=stage.context)
    transport = NcclChainTransport()
    channel = ChainChannel(ChainRole(rank, world), transport, state, device=device)
    capture = BoundaryDigest(rank)

    # Mixed-format startup: the shared chain contract (config, context, the
    # full per-rank format vector, the FP32 boundary dtype, the checkpoint
    # CRC manifest digest) agreed by every rank before any execution.
    from damendaemon.checkpoint_manifest import manifest_digest
    contract = chain_contract(model.config, stage.context, FORMATS,
                              checkpoint_digest=manifest_digest(checkpoint))
    digest = torch.tensor(list(contract), dtype=torch.uint8, device=device)
    local = digest.clone()
    transport.broadcast(digest, 0)
    ack = torch.tensor([int(torch.equal(local, digest))], dtype=torch.int64,
                       device=device)
    transport.all_reduce_min(ack)
    if not ack.item():
        raise SystemExit('chain contract disagreement; aborting before execution')

    if rank == 0:
        from damendaemon.chain_pipeline import ChainPipeline
        pipeline = ChainPipeline(stage, channel, device=device)
        started = time.perf_counter()
        first = pipeline.prefill(SEED, slot=0)
        prefill_s = time.perf_counter() - started
        # ChainPipeline prefill returns the head's token row (a list).
        greedy = [int(first[0])]
        steps = []
        for _ in range(GREEDY - 1):
            mark = time.perf_counter()
            token = int(pipeline.decode_batch([greedy[-1]], [0])[0])
            steps.append(time.perf_counter() - mark)
            greedy.append(token)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        pipeline.close()
        report = dict(mode='mixed-precision-chain', formats=FORMATS, rank=0,
                      checkpoint=checkpoint, layers=layers, bounds=bounds,
                      planned_mixed_split=planned_split, seed=SEED,
                      rank_formats=FORMATS, kernels='triton', context=stage.context,
                      greedy=greedy, prefill_s=round(prefill_s, 3),
                      decode_steps_ms=[round(s * 1000, 1) for s in steps],
                      rank0_max_gib=round(mem, 2))
        target = REPORT_MIXED if 'none' in FORMATS else REPORT_REFERENCE
        target.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    else:
        from damendaemon.chain_pipeline import ChainStageWorker
        worker = ChainStageWorker(stage, channel, device=device)
        name = 'mixed' if 'none' in FORMATS else 'reference'
        # Eager per-round sink: the capture files exist from the first round
        # on, so a teardown stall cannot lose them (final writeback below
        # stays as the authoritative copy).
        capture_hooks(stage, worker, capture,
                      sink=CAPTURES / f'{name}-rank{rank}', bounds=bounds)
        worker.run_forever()
        record = dict(rank=rank, formats=FORMATS, bounds=bounds,
                      boundary_digests=capture.boundaries,
                      head_logits=capture.logits)
        CAPTURES.mkdir(parents=True, exist_ok=True)
        (CAPTURES / f'{name}-rank{rank}.json').write_text(json.dumps(record, indent=2))
        if capture.logit_tensors:
            torch.save(capture.logit_tensors, CAPTURES / f'{name}-rank{rank}-logits.pt')
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return 0

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == '__main__':
    main()

"""#536 gate-4 final: Qwen3.8-27B Q4 over the three-device two-host chain.

The cycle-2..6 machinery (ChainPipeline/ChainStageWorker over NCCL) with the
gate-2-qualified 27B path: Q4 packed, compensated precision, Triton stages,
context 2048. Three ranks across two hosts:
  rank 0: coordinator + layers 0..24  — labnode3 A30 (22528 MiB budget)
  rank 1: layers 24..44               — labnode3 A2  (13312 MiB budget)
  rank 2: layers 44..64 + output head — desktop 3090 (22528 MiB budget)

Layer bounds come from the same choose_stage_splits storage planner the
gate-2 pair used, evaluated with the three per-rank budgets — deterministic,
so every rank derives the identical split [0, 24, 44, 64] (64 layers).

Success gate: 12/12 greedy tokens equal the A30+A2 two-stage in-process
reference of the SAME quantized path (the gate-2 convention; an FP32 CPU
reference does not fit the bounded window — recorded honestly). Timings are
warmup-inclusive and are not serving claims.

Run under torchrun --nnodes=2 --node_rank={0|1} --nproc_per_node={2|1}
--rdzv_backend=static --master_addr=10.30.100.3 --master_port=29500.
"""
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

SNAPSHOT = '/tmp/q3827b'
SEED = list(range(16, 48))


def admit_quantized_context():
    """Bounded chain-probe admission, gate-2 probe convention extended one
    guard: like hybrid_q4_context_probe, only Q4 admission bounds are
    substituted while opt-in, accelerator, architecture, precision and
    vision guards stay intact. Two substitutions, both recorded in the
    evidence as part of what this gate qualifies:
      1. context admission (512 bound -> the probe's 512..32768 window);
      2. the A30-primary pair anchor -> the chain's 3090 tail rank is
         admitted explicitly, because the three-device chain is the first
         Q4 workload on that device (this gate is its qualification).
    Production admission in hybrid_q4_policy is unchanged."""
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
            # Same guards the original enforces; anything failing here also
            # fails production admission.
            original_q4(config, 512)
            raise ValueError('27B chain probe keeps the Q4 opt-in, accelerator, '
                             'architecture, precision and vision guards')
        return None

    def select(model, use_graph, max_batch, kernels):
        from copy import copy as clone
        if model.quantization != 'q4_0':
            raise ValueError('the Q4 chain probe cannot change another weight format')
        validate(model.config, model.max_context)
        admission = clone(model)
        admission.max_context = 512
        return original_select(admission, use_graph, max_batch, kernels)

    hybrid_q4_policy.validate = validate
    hybrid_precision.select = select


admit_quantized_context()

GREEDY = 12
BUDGETS_MIB = [22528, 13312, 22528]


def build_stage(rank, world, device):
    from damendaemon.models import load as load_model
    from damendaemon.hybrid_layout import choose_stage_splits
    from damendaemon.hybrid_stage import HybridLayerStage

    checkpoint = SNAPSHOT if rank in (0, 1) else '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
    model = load_model(checkpoint, 2048, 'q4_0')
    layers = model.config['num_hidden_layers']
    # Same deterministic storage planner as the gate-2 pair; every rank
    # derives the identical three-way split from the same budgets.
    placement = choose_stage_splits(model.weights, model.config, model.max_context,
                                    1, 2, [mib * 1024**2 for mib in BUDGETS_MIB],
                                    scales=model.scales)
    if not placement['storage_plan_found']:
        raise MemoryError('27B three-rank storage plan not found: ' + placement['reason'])
    bounds = [0]
    for budget in placement['stages']:
        bounds.append(budget['layer_end'])
    if len(bounds) != world + 1 or bounds[-1] != layers:
        raise SystemExit(f'planner split {bounds} does not cover {world} ranks/{layers} layers')
    start, end = bounds[rank], bounds[rank + 1]
    stage = HybridLayerStage(model, start, end, max_batch=1, kernels='triton')
    for group in ('w', 'kv', 'conv', 'state'):
        for key, tensor in list(getattr(stage, group).items()):
            getattr(stage, group)[key] = tensor.to(device)
    if hasattr(stage, 'cos'):
        stage.cos = stage.cos.to(device)
        stage.sin = stage.sin.to(device)
        stage.arange = stage.arange.to(device)
    return stage, checkpoint, bounds, layers


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 3:
        raise SystemExit('the 27B chain qualifies exactly 3 ranks')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    os.environ['DAMENDAEMON_HYBRID_PREFILL'] = 'scan'
    # Admission-only budgets: empty_cache() walks len(budgets) CUDA ordinals,
    # so each rank declares only the devices its process can see. The real
    # three-way split is passed explicitly to choose_stage_splits below.
    os.environ['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = (
        '22528' if rank == 2 else '22528,13312')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=1800))
    device = torch.device('cuda', local_rank)

    from damendaemon.stage_protocol import StageState
    from damendaemon.distributed_chain import (ChainChannel, ChainRole,
                                               NcclChainTransport)

    stage, checkpoint, bounds, layers = build_stage(rank, world, device)

    # Explicit metadata context admission (stage_protocol allows up to 2**20);
    # the stage itself was built at model context 2048.
    state = StageState(epoch=1, context=stage.context, slots=stage.max_batch,
                       context_limit=stage.context)
    transport = NcclChainTransport()
    channel = ChainChannel(ChainRole(rank, world), transport, state, device=device)

    report = {'mode': '27b-three-device-chain', 'rank': rank,
              'checkpoint': checkpoint, 'layers': layers, 'bounds': bounds,
              'seed': SEED, 'quantization': 'q4_0', 'precision': 'compensated',
              'kernels': 'triton', 'context': stage.context}

    if rank == 0:
        from damendaemon.chain_pipeline import ChainPipeline
        pipeline = ChainPipeline(stage, channel, device=device)
        started = time.perf_counter()
        pipeline.prefill(SEED, slot=0)
        prefill_s = time.perf_counter() - started
        greedy, step_s = [], []
        token = SEED[-1]
        for _ in range(GREEDY):
            mark = time.perf_counter()
            token = int(pipeline.decode_batch([token], [0])[0])
            step_s.append(time.perf_counter() - mark)
            greedy.append(token)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        report.update({'greedy': greedy, 'prefill_s': round(prefill_s, 3),
                       'decode_steps_ms': [round(s * 1000, 1) for s in step_s],
                       'rank0_max_gib': round(mem, 2), 'failed': pipeline.failed})
        reference = json.loads(Path('/tmp/dd536-27b-reference.json').read_text())
        report['reference_greedy'] = reference['greedy']
        report['matches_reference'] = greedy == reference['greedy']
        Path('/tmp/dd536-27b-chain.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        pipeline.close()
    else:
        from damendaemon.chain_pipeline import ChainStageWorker
        worker = ChainStageWorker(stage, channel, device=device)
        worker.run_forever()

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

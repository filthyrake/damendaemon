"""#536 gate 4 cycle 9: per-token host-transfer cost decomposition.

Two measurements in one torchrun 3-rank run (A30 rank0 coordinator, A2
rank1, desktop 3090 rank2 — the qualified chain topology):

1. Collective microbench at the chain's real payload shapes, comparing
   broadcast origin rank 0 (intra-host labnode3) with origin rank 2
   (cross-host over 10GbE): decode hidden [1, width] fp32, prefill-chunk
   hidden [512, width] fp32, token row int64, and the MIN ack all-reduce.
2. The real 27B Q4 chain decode (context 2048, batch 1, standard
   autoregressive protocol) with per-exchange wall-time instrumentation on
   the coordinator: per-step wall time vs per-step exchange (control+
   transfer) time, with the compute share as the difference.

Timings are wall-clock on synchronized CUDA streams, warmup-inclusive
unless noted; they characterize this topology, not a serving SLO.

Run under torchrun --nnodes=2 --node_rank={0|1} --nproc_per_node={2|1}
--rdzv_backend=static --master_addr=10.30.100.3 --master_port=29500.
Writes /tmp/dd536-transfer-cost.json on the rank-0 host.
"""
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

SEED = list(range(16, 48))
GREEDY = 14
BUDGETS_MIB = [22528, 13312, 22528]
WARMUP_STEPS = 4
MICRO_WARMUP = 10
MICRO_ITERS = 100


def admit_quantized_context():
    """Cycle-7 runner admission, verbatim: probe convention — context
    window and 3090 tail rank substituted; every other guard replicated."""
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


def build_stage(rank, world, device):
    from damendaemon.models import load as load_model
    from damendaemon.hybrid_layout import choose_stage_splits
    from damendaemon.hybrid_stage import HybridLayerStage
    model = load_model('/tmp/q3827b', 2048, 'q4_0')
    layers = model.config['num_hidden_layers']
    placement = choose_stage_splits(model.weights, model.config, model.max_context,
                                    1, 2, [mib * 1024**2 for mib in BUDGETS_MIB],
                                    scales=model.scales)
    bounds = [0] + [b['layer_end'] for b in placement['stages']]
    stage = HybridLayerStage(model, bounds[rank], bounds[rank + 1], max_batch=1,
                             kernels='triton')
    for group in ('w', 'kv', 'conv', 'state'):
        for key, tensor in list(getattr(stage, group).items()):
            getattr(stage, group)[key] = tensor.to(device)
    if hasattr(stage, 'cos'):
        stage.cos = stage.cos.to(device)
        stage.sin = stage.sin.to(device)
        stage.arange = stage.arange.to(device)
    return stage, bounds, layers


def timed_broadcast(transport, tensor, source, iters, warmup, device):
    """Mean wall microseconds per broadcast of `tensor` from `source`."""
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        transport.broadcast(tensor, source)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iters):
        transport.broadcast(tensor, source)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / iters * 1e6


def timed_all_reduce_min(transport, iters, warmup, device):
    ack = torch.ones(1, dtype=torch.int64, device=device)
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        transport.all_reduce_min(ack)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iters):
        transport.all_reduce_min(ack)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / iters * 1e6


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 3:
        raise SystemExit('the transfer-cost decomposition qualifies exactly 3 ranks')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
    os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    os.environ['DAMENDAEMON_HYBRID_PREFILL'] = 'scan'
    os.environ['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = (
        '22528' if rank == 2 else '22528,13312')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=1800))
    device = torch.device('cuda', local_rank)

    admit_quantized_context()

    from damendaemon.stage_protocol import StageState
    from damendaemon.distributed_chain import ChainChannel, ChainRole, NcclChainTransport
    from damendaemon.chain_pipeline import ChainPipeline, ChainStageWorker

    stage, bounds, layers = build_stage(rank, world, device)
    state = StageState(epoch=1, context=stage.context, slots=stage.max_batch,
                       context_limit=stage.context)
    transport = NcclChainTransport()
    channel = ChainChannel(ChainRole(rank, world), transport, state, device=device)

    # --- Part 1: collective microbench at the chain's payload shapes ---
    width = stage.w[next(iter(stage.w))].shape[-1] if stage.w else 5120
    shapes = {
        'decode_hidden_1xW_fp32': torch.zeros((1, width), dtype=torch.float32, device=device),
        'chunk_hidden_512xW_fp32': torch.zeros((512, width), dtype=torch.float32, device=device),
        'token_row_int64': torch.zeros((1,), dtype=torch.int64, device=device),
    }
    micro = {'rank': rank, 'width': width,
             'broadcast_us': {}, 'all_reduce_min_us': None}
    for label, tensor in shapes.items():
        micro['broadcast_us'][label] = {
            'origin_rank0_intra_host': round(timed_broadcast(
                transport, tensor, 0, MICRO_ITERS, MICRO_WARMUP, device), 1),
            'origin_rank2_cross_host': round(timed_broadcast(
                transport, tensor, 2, MICRO_ITERS, MICRO_WARMUP, device), 1),
        }
    micro['all_reduce_min_us'] = round(timed_all_reduce_min(
        transport, MICRO_ITERS, MICRO_WARMUP, device), 1)

    if rank != 0:
        ChainStageWorker(stage, channel, device=device).run_forever()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return

    # --- Part 2: instrumented real decode on the coordinator ---
    # NOTE: a chain broadcast is a synchronization point, so the wall time
    # of an exchange call includes waiting for the SOURCE rank's compute
    # (the src=1 hidden wait is rank 1's layers; the token wait is the
    # head's). Per-call lists are recorded with a round index so decode
    # rounds can be analyzed separately from the prefill round; pure
    # transfer cost is isolated by the microbench above.
    calls = {'exchange_hidden': [], 'exchange_tokens': [], 'finish': []}
    round_marks = []
    current_round = [0]

    def wrap(name, method):
        def timed(*args, **kwargs):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = method(*args, **kwargs)
            torch.cuda.synchronize(device)
            calls[name].append((current_round[0],
                                (time.perf_counter() - started) * 1e6))
            return result
        return timed

    channel.exchange_hidden = wrap('exchange_hidden', channel.exchange_hidden)
    channel.exchange_tokens = wrap('exchange_tokens', channel.exchange_tokens)
    channel.finish = wrap('finish', channel.finish)

    pipeline = ChainPipeline(stage, channel, device=device)
    orig_pipeline_prefill = pipeline.prefill
    orig_pipeline_decode = pipeline.decode_batch

    def round_counting_prefill(*args, **kwargs):
        round_marks.append(('prefill', current_round[0]))
        try:
            return orig_pipeline_prefill(*args, **kwargs)
        finally:
            current_round[0] += 1

    def round_counting_decode(*args, **kwargs):
        round_marks.append(('decode', current_round[0]))
        try:
            return orig_pipeline_decode(*args, **kwargs)
        finally:
            current_round[0] += 1

    pipeline.prefill = round_counting_prefill
    pipeline.decode_batch = round_counting_decode

    pipeline.prefill(SEED, slot=0)
    token = SEED[-1]
    step_ms = []
    all_steps = []
    for step in range(GREEDY):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        token = int(pipeline.decode_batch([token], [0])[0])
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000
        all_steps.append(round(elapsed, 2))
        if step >= WARMUP_STEPS:
            step_ms.append(elapsed)
    steady_ms = sum(step_ms) / len(step_ms)

    prefill_round = round_marks[0][1] if round_marks else 0
    def decode_round_stats(name, rounds_per_step):
        decode_calls = [us for r, us in calls[name] if r != prefill_round]
        if not decode_calls:
            return None
        return {
            'calls_per_step': rounds_per_step,
            'mean_us': round(sum(decode_calls) / len(decode_calls), 1),
            'min_us': round(min(decode_calls), 1),
            'max_us': round(max(decode_calls), 1),
        }
    per_step_exchange_us = None
    decode_hidden = decode_round_stats('exchange_hidden', 2)
    decode_tokens = decode_round_stats('exchange_tokens', 1)
    decode_finish = decode_round_stats('finish', 1)
    if decode_hidden and decode_tokens and decode_finish:
        per_step_exchange_us = (decode_hidden['mean_us'] * decode_hidden['calls_per_step']
                                + decode_tokens['mean_us'] * decode_tokens['calls_per_step']
                                + decode_finish['mean_us'] * decode_finish['calls_per_step'])
    report = {
        'mode': 'transfer-cost-decomposition',
        'topology': {'rank0': 'labnode3 A30', 'rank1': 'labnode3 A2',
                     'rank2': 'desktop 3090 (cross-host 10GbE)'},
        'bounds': bounds, 'layers': layers,
        'microbench': micro,
        'real_decode': {
            'context': 2048, 'steps_total': GREEDY, 'warmup_steps': WARMUP_STEPS,
            'all_steps_ms': all_steps,
            'steady_step_ms': round(steady_ms, 2),
            'decode_round_exchange_waits': {
                'exchange_hidden': decode_hidden,
                'exchange_tokens': decode_tokens,
                'finish': decode_finish,
            },
            'per_step_exchange_wall_us': (round(per_step_exchange_us, 1)
                                          if per_step_exchange_us is not None else None),
            'per_step_exchange_wall_ms': (round(per_step_exchange_us / 1000, 2)
                                          if per_step_exchange_us is not None else None),
            'note': 'decode-round means only (the prefill round is excluded). '
                    'An exchange wall time includes waiting for the SOURCE '
                    'rank compute (broadcasts are sync points) — it is an '
                    'upper bound on transfer, not pure transfer; the microbench '
                    'isolates pure collective cost. Warmup-inclusive except '
                    'steady_step_ms (first 4 steps excluded)',
        },
    }
    Path('/tmp/dd536-transfer-cost.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    pipeline.close()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

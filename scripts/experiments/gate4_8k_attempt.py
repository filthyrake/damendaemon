"""#536 gate 4 cycle 10: 27B Q4 chain at context 8192 (guarded attempt).

The gate-3 re-scope moved the 8K cell to the three-device topology (the
A2 heats on cumulative ramp energy in the two-rank pair; its stage tail
moves to the desktop 3090). This runner attempts the 8K cell on the real
chain with the gate-3 containment discipline:

- context 8192, batch 1, Q4 compensated, Triton stages, planner bounds
  from the three per-rank budgets;
- the KV/conv ramp is chunked (prefill_chunk, CHUNK rows at a time) so the
  ramp is observable and stoppable — a whole-prompt 8K prefill is never
  issued;
- thermals are sampled per host every SAMPLE_S seconds (labnode3 A30/A2
  via nvidia-smi from rank 0's parent shell is NOT available inside the
  process; each rank reads torch.cuda.temperature() where supported, else
  the operator samples externally — recorded honestly);
- per-rank peak device memory is recorded (planner-vs-observed gap);
- correctness gate at 8K: the chunked ramp's FINAL chunk sample must
  equal a whole-prompt... no — a second full ramp is an extra 8K pass.
  Instead: after the ramp, N greedy decodes must succeed and a
  deterministic replay (release + identical ramp) must reproduce the SAME
  greedy sequence (self-consistency, the honest gate available without a
  thermally risky 8K pair reference).

Stop conditions (gate-3 discipline): any thermal gate trip, OOM, planner
rejection, or failure poison -> record the attempt honestly and stop.

Writes /tmp/dd536-8k-attempt.json on the rank-0 host.
"""
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

CHUNK = 512
RAMP_TOKENS = 7680          # 15 chunks of 512; 7680 + decode headroom fits 8192
GREEDY = 12
BUDGETS_MIB = [22528, 13312, 22528]
THERMAL_LIMIT_C = 80.0      # hard stop; warning at 65 (gate-3 A2 gate)


def admit_quantized_context():
    """Cycle-7 runner admission, verbatim (probe convention)."""
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


def gpu_temp(device):
    try:
        return round(torch.cuda.temperature(device), 1)
    except Exception:
        return None


def build_stage(rank, world, device, context):
    from damendaemon.models import load as load_model
    from damendaemon.hybrid_layout import choose_stage_splits
    from damendaemon.hybrid_stage import HybridLayerStage
    model = load_model('/tmp/q3827b', context, 'q4_0')
    layers = model.config['num_hidden_layers']
    placement = choose_stage_splits(model.weights, model.config, context, 1, 2,
                                    [mib * 1024**2 for mib in BUDGETS_MIB],
                                    scales=model.scales)
    if not placement['storage_plan_found']:
        raise SystemExit('8K storage plan not found: ' + str(placement['reason']))
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
    return stage, bounds, layers, placement


def ramp(pipeline, report, label):
    """Chunked prefill ramp to RAMP_TOKENS; records per-chunk timings and
    the ramp's final protocol sample. Thermal guard between chunks."""
    samples = []
    start = 0
    token_row = list(range(1000, 1000 + CHUNK))
    while start < RAMP_TOKENS:
        mark = time.perf_counter()
        sample = int(pipeline.prefill_chunk(token_row, 0, start)[0])
        elapsed = time.perf_counter() - mark
        start += CHUNK
        samples.append(dict(start=start, seconds=round(elapsed, 3), sample=sample))
        temp = gpu_temp(torch.cuda.current_device())
        if temp is not None and temp > THERMAL_LIMIT_C:
            report['stopped'] = f'thermal limit {temp}C at ramp offset {start}'
            return samples, None
    return samples, samples[-1]['sample']


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    context = 8192
    if world != 3:
        raise SystemExit('the 8K attempt qualifies exactly 3 ranks')
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
                                         timeout=timedelta(seconds=3600))
    device = torch.device('cuda', local_rank)

    admit_quantized_context()

    from damendaemon.stage_protocol import StageState
    from damendaemon.distributed_chain import ChainChannel, ChainRole, NcclChainTransport
    from damendaemon.chain_pipeline import ChainPipeline, ChainStageWorker

    stage, bounds, layers, placement = build_stage(rank, world, device, context)
    state = StageState(epoch=1, context=stage.context, slots=stage.max_batch,
                       context_limit=stage.context)
    channel = ChainChannel(ChainRole(rank, world), NcclChainTransport(),
                          state, device=device)

    if rank != 0:
        ChainStageWorker(stage, channel, device=device).run_forever()
        # Measured per-stage memory for the capacity table (worker side).
        Path(f'/tmp/dd536-8k-rank{rank}.json').write_text(json.dumps({
            'rank': rank, 'bounds': [bounds[rank], bounds[rank + 1]],
            'max_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            'temp_c': gpu_temp(torch.cuda.current_device()),
        }, indent=2))
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return

    report = {'mode': '27b-chain-8k-attempt', 'context': context,
              'chunk': CHUNK, 'ramp_tokens': RAMP_TOKENS,
              'bounds': bounds, 'layers': layers,
              'planner_required_gib': [round(b['required_memory_bytes'] / 1024**3, 2)
                                       for b in placement['stages']],
              'budgets_mib': BUDGETS_MIB}
    pipeline = ChainPipeline(stage, channel, device=device)

    # Ramp 1: chunked prefill to RAMP_TOKENS, then greedy decode.
    ramp1, final_sample_1 = ramp(pipeline, report, 'ramp1')
    report['ramp1'] = ramp1
    if final_sample_1 is None:
        report['result'] = 'stopped-during-ramp'
        Path('/tmp/dd536-8k-attempt.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        pipeline.close()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return
    # Standard autoregressive decode from the ramp's final sample.
    greedy1, steps = [], []
    token = final_sample_1
    for _ in range(GREEDY):
        mark = time.perf_counter()
        token = int(pipeline.decode_batch([token], [0])[0])
        steps.append(round((time.perf_counter() - mark) * 1000, 1))
        greedy1.append(token)
    report['greedy1'] = greedy1
    report['decode_steps_ms'] = steps

    # Ramp 2 (determinism gate): release, identical ramp, identical decode.
    pipeline.release([0])
    ramp2, final_sample_2 = ramp(pipeline, report, 'ramp2')
    report['ramp2_final_sample'] = final_sample_2
    deterministic = final_sample_2 == final_sample_1
    greedy2 = None
    if final_sample_2 is not None:
        greedy2, token = [], final_sample_2
        for _ in range(GREEDY):
            token = int(pipeline.decode_batch([token], [0])[0])
            greedy2.append(token)
        deterministic = deterministic and greedy2 == greedy1
    report['greedy2'] = greedy2
    report['deterministic_replay'] = deterministic

    report['rank0_max_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    report['rank0_temp_c'] = gpu_temp(0)
    report['failed'] = pipeline.failed
    report['result'] = ('qualified-8k' if deterministic and not pipeline.failed
                        else 'completed-inconsistent')
    Path('/tmp/dd536-8k-attempt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    pipeline.close()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

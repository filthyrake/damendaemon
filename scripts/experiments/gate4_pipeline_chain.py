"""#536 gate-4 cycle 2b: qualify ChainPipeline/ChainStageWorker on the real
three-rank NCCL chain (labnode3 A30 + A2, desktop 3090).

Rank 0 runs the ChainPipeline coordinator; ranks 1-2 run ChainStageWorker.
Success gate: 12/12 greedy tokens equal the single-rank reference report
(/tmp/dd536-g4-reference.json, kernels='torch').

Run under torchrun --nnodes=2 --node_rank={0|1} --nproc_per_node={2|1}
--rdzv_backend=static --master_addr=10.30.100.3 --master_port=29500
"""
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

SEED_TOKENS = list(range(16, 48))
GREEDY_TOKENS = 12


def build_stage(checkpoint: str, rank: int, world: int, device):
    from damendaemon.models import load as load_model
    model = load_model(checkpoint, 512, None)
    layers = model.config['num_hidden_layers']
    bounds = [round(i * layers / world) for i in range(world + 1)]
    start, end = bounds[rank], bounds[rank + 1]
    from damendaemon.hybrid_stage import HybridLayerStage
    stage = HybridLayerStage(model, start, end, max_batch=1, kernels='torch')
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


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world not in (2, 3):
        raise SystemExit('gate-4 pipeline qualifies 2 or 3 ranks')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=120))
    device = torch.device('cuda', local_rank)

    from damendaemon.stage_protocol import StageState
    from damendaemon.distributed_chain import (ChainChannel, ChainRole,
                                               NcclChainTransport)
    from damendaemon.chain_pipeline import ChainPipeline, ChainStageWorker

    checkpoint = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/'
                  'snapshots/2fc06364715b967f1860aea9cf38778875588b17')
    if os.environ.get('G4_CHECKPOINT'):
        checkpoint = os.environ['G4_CHECKPOINT']

    stage, bounds = build_stage(checkpoint, rank, world, device)
    state = StageState(epoch=1, context=stage.context, slots=stage.max_batch)
    transport = NcclChainTransport()
    channel = ChainChannel(ChainRole(rank, world), transport, state, device=device)

    report = {'mode': 'pipeline-chain', 'checkpoint': checkpoint, 'rank': rank,
              'bounds': bounds, 'seed': SEED_TOKENS, 'kernels': 'torch'}

    if rank == 0:
        pipeline = ChainPipeline(stage, channel, device=device)
        started = time.perf_counter()
        pipeline.prefill(SEED_TOKENS, slot=0)
        prefill_s = time.perf_counter() - started
        greedy = []
        step_s = []
        token = SEED_TOKENS[-1]
        for _ in range(GREEDY_TOKENS):
            mark = time.perf_counter()
            token = int(pipeline.decode_batch([token], slots=[0])[0])
            step_s.append(time.perf_counter() - mark)
            greedy.append(token)
        report.update({'greedy': greedy, 'prefill_s': round(prefill_s, 4),
                       'decode_steps_ms': [round(s * 1000, 2) for s in step_s],
                       'failed': pipeline.failed})
        reference = json.loads(Path('/tmp/dd536-g4-reference.json').read_text())
        report['reference_greedy'] = reference['greedy']
        report['matches_reference'] = greedy == reference['greedy']
        Path('/tmp/dd536-g4-pipeline.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        pipeline.close()
    else:
        worker = ChainStageWorker(stage, channel, device=device)
        worker.run_forever()

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

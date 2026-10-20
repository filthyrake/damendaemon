"""Experimental chain HTTP service: three-rank greedy serving under torchrun.

Ranks 1..head run ChainStageWorker loops; rank 0 serves /v1/completions
through the proven DistributedRuntime scheduler over ChainServingPair.
Model identity is fingerprinted on EVERY rank and agreed via the chain
transport before serving — a mismatched checkpoint aborts startup.

Run: torchrun --nnodes=2 --node_rank={0|1} --nproc_per_node={2|1} \
        --rdzv_backend=static --master_addr=<rank0 host> --master_port=29500 \
        -m damendaemon chain-serve <checkpoint> --port 8017
"""
import argparse
import os
import signal
import threading
from datetime import timedelta

import torch


def build_stage(checkpoint, rank, world, device, context, quantization, *, slots):
    from damendaemon.models import load as load_model
    from damendaemon.hybrid_stage import HybridLayerStage
    model = load_model(checkpoint, context, quantization)
    layers = model.config['num_hidden_layers']
    bounds = [round(i * layers / world) for i in range(world + 1)]
    stage = HybridLayerStage(model, bounds[rank], bounds[rank + 1],
                             max_batch=slots, kernels='torch')
    for group in ('w', 'kv', 'conv', 'state'):
        for key, tensor in list(getattr(stage, group).items()):
            getattr(stage, group)[key] = tensor.to(device)
    if hasattr(stage, 'cos'):
        stage.cos = stage.cos.to(device)
        stage.sin = stage.sin.to(device)
        stage.arange = stage.arange.to(device)
    return model, stage, [[bounds[i], bounds[i + 1]] for i in range(world)]


def context_slots(context):
    return 4 if context >= 2048 else 2


def main(argv=None):
    p = argparse.ArgumentParser(prog='damendaemon chain-serve',
        description='Experimental three-rank chain greedy HTTP service under torchrun')
    p.add_argument('path', help='matching local safetensors snapshot on every rank')
    p.add_argument('--quantization', choices=('none', 'q8_0', 'q4_0'),
                   default=os.environ.get('DAMENDAEMON_QUANTIZATION', 'none'))
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--alias', default='model')
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--batch-size', type=int, choices=range(1, 5), default=2)
    p.add_argument('--model', action='append', default=[], metavar='ALIAS=PATH',
                   help='one additional compatible local model on every rank for resident switching (unquantized only)')
    p.add_argument('--queue-size', type=int, default=8)
    p.add_argument('--queue-timeout', type=float, default=120)
    p.add_argument('--idle-timeout', type=float, default=0,
                   help='optional rank-0 shutdown after idle seconds; zero disables')
    p.add_argument('--chunk-size', type=int, default=128,
                   help='chain prefill chunk rows for interleaved prefill')
    p.add_argument('--interleaved-prefill', action='store_true',
                   help='chunk long prompts across decode rounds instead of whole-prompt prefill')
    p.add_argument('--client-prefill-turns', action='store_true',
                   help='weighted per-client prefill chunk shares (requires --interleaved-prefill)')
    p.add_argument('--bind', default='127.0.0.1')
    args = p.parse_args(argv)
    quantization = None if args.quantization == 'none' else args.quantization
    if not 1 <= args.context <= 512 or not 0 <= args.port <= 65535:
        p.error('invalid context or port')
    if not 0 <= args.idle_timeout <= 86400 or not 1 <= args.queue_size <= 64:
        p.error('invalid idle timeout or queue size')
    if not 0 < args.queue_timeout <= 300 or not 1 <= len(args.alias) <= 128:
        p.error('invalid queue timeout or model alias')
    if args.batch_size > context_slots(args.context):
        p.error('batch size exceeds the context-sized slot table')
    if not 1 <= args.chunk_size <= 4096 or args.chunk_size > args.context:
        p.error('chunk size must be 1..4096 and fit the context')
    if args.client_prefill_turns and not args.interleaved_prefill:
        p.error('client prefill turns require --interleaved-prefill')
    if len(args.model) > 1 or any('=' not in entry for entry in args.model):
        p.error('at most one additional ALIAS=PATH model is supported')
    additional = [tuple(entry.split('=', 1)) for entry in args.model]
    if any(not alias or not path or alias == args.alias for alias, path in additional):
        p.error('additional model needs a distinct alias and local path')
    if additional and quantization is not None:
        p.error('resident chain switching currently requires unquantized models')

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if not 2 <= world <= 16:
        raise SystemExit('chain serving requires 2..16 ranks')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=120))
    device = torch.device('cuda', local_rank)

    from damendaemon.distributed_chain import ChainChannel, ChainRole, NcclChainTransport
    from damendaemon.stage_identity import fingerprint
    from damendaemon.stage_protocol import StageState

    model, stage, layer_bounds = build_stage(args.path, rank, world, device,
                                             args.context, quantization,
                                             slots=args.batch_size)
    # Additional resident models: same decoder config, distinct identity.
    from damendaemon.models import load as load_model
    extra_models = []
    for alias, path in additional:
        extra = load_model(path, args.context, quantization)
        if extra.config != model.config:
            raise SystemExit(f'chain model {alias} has an incompatible decoder config')
        extra_models.append((alias, extra))
    # Fingerprint BEFORE moving weights: identity requires prepared CPU
    # tensors, and every rank must agree before rank 0 serves.
    transport = NcclChainTransport()

    def agree(alias, fingerprinted):
        digest = torch.tensor(list(fingerprinted), dtype=torch.uint8, device=device)
        local = digest.clone()
        transport.broadcast(digest, 0)  # rank 0 publishes; every rank receives
        ack = torch.tensor([int(torch.equal(local, digest))], dtype=torch.int64,
                           device=device)
        transport.all_reduce_min(ack)
        if not ack.item():
            raise SystemExit(f'chain model identity disagreement on {alias}; '
                             'aborting before serving')

    identity = fingerprint(model)
    agree(args.alias, identity)
    extra_identities = []
    for alias, extra in extra_models:
        extra_identity = fingerprint(extra)
        agree(alias, extra_identity)
        extra_identities.append((alias, extra_identity))

    # Per-rank shard table: index 0 is the resident (initial) model; each
    # entry is the rank's layer-range weight dict of CPU tensors.
    shards = [{name: source[name] for name in stage.w}
              for source in (model.weights, *(extra.weights for _, extra in extra_models))]

    state = StageState(epoch=1, context=args.context, slots=args.batch_size,
                       models=1 + len(extra_models))
    channel = ChainChannel(ChainRole(rank, world), transport, state, device=device)

    if rank != 0:
        from damendaemon.chain_pipeline import ChainStageWorker
        ChainStageWorker(stage, channel, device=device, shards=shards).run_forever()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        return 0

    from damendaemon.chain_pipeline import ChainPipeline
    from damendaemon.chain_serving import (ChainDistributedRuntime, ChainServingPair,
                                           handler_for_chain)
    from damendaemon.server import BoundedHTTPServer
    pipeline = ChainPipeline(stage, channel, device=device, shards=shards)
    models = {args.alias: model}
    identities = {args.alias: identity}
    for alias, extra in extra_models:
        models[alias] = extra
    for (alias, extra_identity) in extra_identities:
        identities[alias] = extra_identity
    pair = ChainServingPair(pipeline, model, alias=args.alias,
                            identity=identity, stage_layers=layer_bounds,
                            chunk_size=args.chunk_size,
                            models=models, identities=identities)
    runtime = ChainDistributedRuntime(pair, args.alias, args.queue_size,
                                      args.queue_timeout,
                                      interleaved_prefill=args.interleaved_prefill,
                                      client_prefill_turns=args.client_prefill_turns)
    base = handler_for_chain(runtime)

    class Handler(base):
        def do_GET(self):
            if self.path == '/health':
                stopped = runtime.closed.is_set()
                self.send_json(503 if stopped else 200,
                               dict(status='closed' if stopped else 'ok'))
            else:
                super().do_GET()

    server = BoundedHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = False
    server.block_on_close = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = signal.signal(signal.SIGTERM,
                              lambda number, frame: runtime.close())
    print(f'damendaemon chain listening on http://{args.bind}:{server.server_port}',
          flush=True)
    try:
        runtime.run(idle_timeout=args.idle_timeout)
    except KeyboardInterrupt:
        runtime.close()
    finally:
        server.shutdown(); server.server_close()
        runtime.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0

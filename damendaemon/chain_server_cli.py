"""Experimental chain HTTP service: three-rank greedy serving under torchrun.

Ranks 1..head run ChainStageWorker loops; rank 0 serves /v1/completions
through the proven DistributedRuntime scheduler over ChainServingPair.
Model identity is fingerprinted on EVERY rank and agreed via the chain
transport before serving — a mismatched checkpoint aborts startup.

Quantized chain serving (--quantization q4_0, #536 gate 4): every rank sets
the explicit experimental pipeline/Q4 opt-ins plus the distributed-chain
Q4 opt-in (DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN=1), builds Triton
stages, and derives its layer bounds from the same choose_stage_splits
storage planner as the gate-2 pair, evaluated with the per-rank budgets
(--rank-budgets-mib, one entry per rank, identical on every rank). The
admission budgets env each rank process declares is its NODE-LOCAL slice:
empty_cache() walks one CUDA ordinal per entry, so a rank may only declare
the devices its process can see.

Mixed per-rank precision (--quantization q4_0,q4_0,none, #601): one format
entry per rank, identical on every rank, from {none, q4_0}. Each rank loads
the checkpoint in its own format (Q4 ranks keep the compensated chain
admission; unquantized ranks run the qualified FP16 Triton path), every
rank prices every other rank's stage from a format-independent table so
the split stays deterministic, the startup digest becomes the shared
chain contract (config, context, the full format vector, the FP32 boundary
dtype) instead of loaded weight bytes, and /v1/models/status reports the
per-rank formats and weight bytes. Q8_0 participation stays rejected with
its policy reason.

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


def build_stage(checkpoint, rank, world, device, context, quantization, *, slots,
                rank_budgets=None, rank_quantizations=None):
    """Build this rank's stage; quantization is per-rank for mixed chains (#601).

    Uniform chains (rank_quantizations is None) keep the exact existing
    behavior. Mixed chains load THIS rank's format, price every rank's
    stage from the format-independent pricing table (so every rank derives
    the identical split) and build Triton stages on every rank.
    """
    from damendaemon.models import load as load_model
    from damendaemon.hybrid_stage import HybridLayerStage
    # Mixed chains pass the explicit per-rank list; uniform chains keep the
    # exact existing single-quantization path (tensor planner, real loads).
    if rank_quantizations is not None:
        local = rank_quantizations[rank]
        local_quantization = None if local == 'none' else local
    else:
        local_quantization = quantization
    model = load_model(checkpoint, context, local_quantization)
    layers = model.config['num_hidden_layers']
    if rank_budgets is None:
        bounds = [round(i * layers / world) for i in range(world + 1)]
        kernels = 'torch'
    else:
        bounds = None
        if rank_quantizations is None:
            # Same deterministic storage planner as the gate-2 pair, evaluated
            # with the per-rank budgets; every rank derives the identical split.
            from damendaemon.hybrid_layout import choose_stage_splits
            placement = choose_stage_splits(model.weights, model.config,
                                           model.max_context, slots, 2,
                                           [mib * 1024**2 for mib in rank_budgets],
                                           scales=getattr(model, 'scales', None))
        else:
            # Per-rank formats: every rank prices every rank's stage in that
            # rank's format from its own load's format-independent table.
            from damendaemon.hybrid_layout import (choose_stage_splits_formats,
                                                   stage_pricing_table)
            pricing = stage_pricing_table(model.weights, getattr(model, 'scales', None),
                                          model.config)
            placement = choose_stage_splits_formats(
                pricing, model.config, model.max_context, slots, 2,
                [mib * 1024**2 for mib in rank_budgets], rank_quantizations)
        if not placement['storage_plan_found']:
            raise SystemExit('chain storage plan not found for the given '
                             'rank budgets: ' + str(placement['reason']))
        bounds = [0]
        for budget in placement['stages']:
            bounds.append(budget['layer_end'])
        if len(bounds) != world + 1 or bounds[-1] != layers:
            raise SystemExit(f'planner split {bounds} does not cover {world} '
                             f'ranks/{layers} layers')
        kernels = 'triton'
    stage = HybridLayerStage(model, bounds[rank], bounds[rank + 1],
                             max_batch=slots, kernels=kernels)
    for group in ('w', 'kv', 'conv', 'state'):
        for key, tensor in list(getattr(stage, group).items()):
            getattr(stage, group)[key] = tensor.to(device)
    if hasattr(stage, 'cos'):
        stage.cos = stage.cos.to(device)
        stage.sin = stage.sin.to(device)
        stage.arange = stage.arange.to(device)
    return model, stage, [[bounds[i], bounds[i + 1]] for i in range(world)]


def node_local_budgets(rank, local_rank, local_size, rank_budgets):
    """The budgets entries a rank's PROCESS may declare (admission env).

    empty_cache() walks one CUDA ordinal per DAMENDAEMON_HYBRID_PIPELINE_
    BUDGETS_MIB entry, so a rank process may only declare the devices it
    can see — its node's slice of the per-rank budget list. The full list
    still drives the layer-split planner (passed explicitly, not via env).
    """
    start = rank - local_rank
    if (type(local_size) is not int or local_size < 1
            or start < 0 or start + local_size > len(rank_budgets)):
        raise ValueError('node-local budget slice is outside the rank budgets')
    return rank_budgets[start:start + local_size]


def context_slots(context):
    return 4 if context >= 2048 else 2


def main(argv=None):
    p = argparse.ArgumentParser(prog='damendaemon chain-serve',
        description='Experimental three-rank chain greedy HTTP service under torchrun')
    p.add_argument('path', help='matching local safetensors snapshot on every rank')
    p.add_argument('--quantization',
                   default=os.environ.get('DAMENDAEMON_QUANTIZATION', 'none'),
                   help='uniform weight format (none/q8_0/q4_0), or a comma-separated '
                        'per-rank list from none/q4_0 (mixed chains, #601; '
                        'q8_0 chain participation stays rejected)')
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
    p.add_argument('--rank-budgets-mib', metavar='MIB,...',
                   help='per-rank storage budgets in MiB, one entry per rank, identical on '
                        'every rank (required for --quantization q4_0; drives the layer split)')
    p.add_argument('--bind', default='127.0.0.1')
    args = p.parse_args(argv)
    quantization = None if args.quantization == 'none' else args.quantization
    rank_quantizations = None
    if ',' in args.quantization:
        # Per-rank heterogeneous precision (#601): one entry per rank,
        # identical on every rank, mixed across ranks.
        entries = args.quantization.split(',')
        if (not 2 <= len(entries) <= 16
                or any(entry not in ('none', 'q8_0', 'q4_0') for entry in entries)):
            p.error('per-rank --quantization needs 2..16 comma-separated entries '
                    'from none, q4_0, q8_0')
        if 'q8_0' in entries:
            p.error('Q8_0 chain participation stays rejected: compensated Q8 is '
                    'ROCm-scoped and no Q8 chain cell is qualified (admission '
                    'is honest, not silently downgraded)')
        if len(set(entries)) < 2 or 'q4_0' not in entries:
            p.error('mixed per-rank --quantization needs at least one q4_0 rank '
                    'and one differing rank (uniform lists use the single-value form)')
        rank_quantizations = entries
        quantization = None
    rank_budgets = None
    if args.rank_budgets_mib is not None:
        entries = args.rank_budgets_mib.split(',')
        if not entries or any(not entry.isascii() or not entry.isdecimal()
                              or int(entry) <= 0 for entry in entries):
            p.error('rank budgets are comma-separated positive MiB integers')
        rank_budgets = [int(entry) for entry in entries]
    quantized_chain = quantization == 'q4_0' or rank_quantizations is not None
    context_bound = 2048 if quantized_chain else 512
    if not 1 <= args.context <= context_bound or not 0 <= args.port <= 65535:
        p.error(f'invalid context or port (context 1..{context_bound} for '
                f'--quantization {args.quantization})')
    if quantization == 'q8_0':
        p.error('quantized chain serving currently supports q4_0 only '
                '(the #536 gate-4 qualified path)')
    if quantization == 'q4_0' and rank_budgets is None:
        p.error('--quantization q4_0 requires --rank-budgets-mib (one entry per rank)')
    if rank_quantizations is not None and rank_budgets is None:
        p.error('per-rank --quantization requires --rank-budgets-mib (one entry per rank)')
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
    if additional and (quantization is not None or rank_quantizations is not None):
        p.error('resident chain switching currently requires unquantized models')

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if not 2 <= world <= 16:
        raise SystemExit('chain serving requires 2..16 ranks')
    if rank_budgets is not None and len(rank_budgets) != world:
        raise SystemExit(f'--rank-budgets-mib needs exactly {world} entries '
                         f'(got {len(rank_budgets)})')
    if rank_quantizations is not None and len(rank_quantizations) != world:
        raise SystemExit(f'per-rank --quantization needs exactly {world} entries '
                         f'(got {len(rank_quantizations)})')
    os.environ.setdefault('MASTER_ADDR', '10.30.100.3')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['DAMENDAEMON_DEVICE'] = 'cuda'
    if quantized_chain:
        # The explicit experimental chain-Q4 admission (#536 gate 4): the
        # opt-ins mirror the qualified runner; every guard except the
        # A30-primary anchor and the 512 context bound still applies. For a
        # mixed chain (#601) each rank declares its OWN format's opt-ins:
        # Q4 ranks keep the compensated chain admission, unquantized ranks
        # run the qualified FP16 Triton pipeline path.
        local_size = int(os.environ['LOCAL_WORLD_SIZE'])
        local = node_local_budgets(rank, int(os.environ.get('LOCAL_RANK', 0)),
                                    local_size, rank_budgets)
        if torch.cuda.device_count() < len(local):
            raise SystemExit(f'rank {rank} sees {torch.cuda.device_count()} CUDA '
                             f'devices but its node-local budgets declare {len(local)}')
        rank_format = (rank_quantizations[rank] if rank_quantizations is not None
                       else 'q4_0')
        os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE'] = '1'
        os.environ['DAMENDAEMON_HYBRID_PIPELINE_BUDGETS_MIB'] = ','.join(str(m) for m in local)
        if rank_format == 'q4_0':
            os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT'] = '1'
            os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
            os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN'] = '1'
            os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
        else:
            os.environ['DAMENDAEMON_HYBRID_PRECISION'] = 'float16'
        os.environ['DAMENDAEMON_HYBRID_PREFILL'] = 'scan'
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world,
                                         timeout=timedelta(seconds=1800))
    device = torch.device('cuda', local_rank)

    from damendaemon.distributed_chain import ChainChannel, ChainRole, NcclChainTransport
    from damendaemon.stage_identity import chain_contract, fingerprint
    from damendaemon.stage_protocol import StageState

    model, stage, layer_bounds = build_stage(args.path, rank, world, device,
                                             args.context, quantization,
                                             slots=args.batch_size,
                                             rank_budgets=rank_budgets,
                                             rank_quantizations=rank_quantizations)
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

    if rank_quantizations is None:
        identity = fingerprint(model)
    else:
        # Mixed-format chains cannot compare loaded weight bytes (each rank
        # loads its own format); the shared chain contract pins config,
        # context, the full per-rank format vector and the FP32 boundary
        # dtype every rank derives identically from its own load, PLUS the
        # checkpoint's CRC manifest when staged with one (crc32.txt, the
        # gate-4 discipline) — without it a wrong checkpoint with identical
        # architecture cannot be detected across ranks.
        from damendaemon.checkpoint_manifest import manifest_digest
        identity = chain_contract(model.config, args.context, rank_quantizations,
                                  checkpoint_digest=manifest_digest(args.path))
    agree(args.alias, identity)
    extra_identities = []
    for alias, extra in extra_models:
        extra_identity = fingerprint(extra)
        agree(alias, extra_identity)
        extra_identities.append((alias, extra_identity))

    # Per-rank shard table: index 0 is the resident (initial) model; each
    # entry is the rank's layer-range weight dict of CPU tensors. Quantized
    # and mixed-format chain serving keep a single resident model (no
    # SWITCH support), so no shard table is staged.
    if quantization is None and rank_quantizations is None:
        shards = [{name: source[name] for name in stage.w}
                  for source in (model.weights, *(extra.weights for _, extra in extra_models))]
    else:
        shards = None

    # Truthful per-rank precision/weight accounting for mixed chains (#601):
    # every rank's stage bytes priced in that rank's format from this rank's
    # own format-independent table (the same table that derived the split).
    rank_weight_bytes = None
    if rank_quantizations is not None:
        from damendaemon.hybrid_layout import stage_budget_formats, stage_pricing_table
        pricing = stage_pricing_table(model.weights, getattr(model, 'scales', None),
                                      model.config)
        rank_weight_bytes = []
        for index, (start, end) in enumerate(layer_bounds):
            budget = stage_budget_formats(pricing, model.config, args.context,
                                          args.batch_size, 2, start, end,
                                          rank_quantizations[index])
            rank_weight_bytes.append(budget['weight_bytes'] + budget['scale_bytes'])

    state = StageState(epoch=1, context=args.context, slots=args.batch_size,
                       models=1 + len(extra_models), context_limit=args.context)
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
                            models=models, identities=identities,
                            stage_quantizations=rank_quantizations,
                            stage_weight_bytes=rank_weight_bytes)
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
    print(f'damendaemon chain listening on http://{args.bind}:{server.server_port}', flush=True)
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

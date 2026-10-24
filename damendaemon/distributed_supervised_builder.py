"""Real (GPU) runtime builder for the supervised distributed service.

Runs in the coordinator process. Each generation owns a fresh rendezvous
store port and run id; the worker child is spawned with the matching torchrun
environment, then the coordinator constructs StagePair/DistributedRuntime
against the same store. CUDA must not be initialized before the supervisor
sets CUDA_VISIBLE_DEVICES to both explicit GPU UUIDs.
"""
import os
import subprocess
import sys
from datetime import timedelta

WORKER_SESSION = 'distributed-serve'


def gpu_uuids(torch, count=2):
    uuids = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, 'uuid', None)
        if uuid is None:
            raise ValueError('supervised distributed-serve needs torch with exposed device UUIDs')
        uuids.append('GPU-' + str(uuid))
    if len(set(uuids)) != count:
        raise ValueError('supervised distributed-serve needs distinct GPU UUIDs')
    return uuids


def supervised_builder(args, torch):
    """Return runtime_builder(generation) -> (pair, runtime) for the supervisor."""
    uuids = gpu_uuids(torch)
    additional = [tuple(entry.split('=', 1)) for entry in args.model]
    executable = sys.executable

    def builder(generation):
        import socket
        import torch.distributed as dist
        from .distributed import StagePair
        from .distributed_serving import DistributedRuntime
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            rendezvous_port = probe.getsockname()[1]
        run_id = f'supervised-{os.getpid()}-{generation}'
        store = dist.TCPStore('127.0.0.1', rendezvous_port, world_size=None, is_master=True,
                              wait_for_workers=False, timeout=timedelta(seconds=30))
        common = dict(WORLD_SIZE='2', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(rendezvous_port),
                      TORCHELASTIC_USE_AGENT_STORE='true', TORCHELASTIC_RUN_ID=run_id,
                      TORCHELASTIC_RESTART_COUNT='0', DAMENDAEMON_DEVICE='cuda',
                      DAMENDAEMON_QUANTIZATION=args.quantization)
        child = dict(os.environ)
        child.update(common, RANK='1', LOCAL_RANK='1',
                     CUDA_VISIBLE_DEVICES=','.join(uuids))
        worker_command = [executable, '-m', 'damendaemon.distributed_worker', args.path,
                          '--context', str(args.context), '--quantization', args.quantization,
                          '--alias', args.alias, '--batch-size', str(args.batch_size)]
        for entry in args.model:
            worker_command += ['--model', entry]
        worker = subprocess.Popen(worker_command, env=child, start_new_session=True)
        parent = dict(os.environ)
        parent.update(common, RANK='0', LOCAL_RANK='0',
                      CUDA_VISIBLE_DEVICES=','.join(uuids))
        os.environ.clear(); os.environ.update(parent)
        try:
            pair = StagePair(args.path, args.context, args.quantization, lambda model: [],
                             session=WORKER_SESSION, max_batch=args.batch_size,
                             model_alias=args.alias, additional_models=additional)
            if pair.rank != 0:
                raise ValueError('the supervisor coordinator requires RANK=0')
            runtime = DistributedRuntime(pair, args.alias, args.queue_size,
                                         args.queue_timeout,
                                         client_decode_turns=args.client_decode_turns)
        except BaseException:
            worker.terminate()
            raise
        pair.supervised_worker = worker
        pair.supervised_store = store
        return pair, runtime
    return builder


def teardown_supervised_pair(pair):
    """Generation teardown: kill the worker child, close store and pair."""
    worker = getattr(pair, 'supervised_worker', None)
    if worker is not None and worker.poll() is None:
        worker.terminate()
        try:worker.wait(timeout=10)
        except subprocess.TimeoutExpired:worker.kill()
    store = getattr(pair, 'supervised_store', None)
    try:
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    except Exception:pass
    try:
        if store is not None:store.close()
    except Exception:pass

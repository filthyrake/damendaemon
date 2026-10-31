"""Real (GPU) runtime builder for the supervised distributed service.

Runs in the coordinator process. Each generation owns a fresh rendezvous
store port and run id; the worker child is spawned with the matching torchrun
environment, then the coordinator constructs StagePair/DistributedRuntime
against the same store. CUDA must not be initialized before the supervisor
sets CUDA_VISIBLE_DEVICES to both explicit GPU UUIDs.

Cross-host mode (--worker-host): the rank-1 worker runs over SSH on a remote
host with one CUDA device; the rendezvous store binds the coordinator's LAN
address and each side pins NCCL_SOCKET_IFNAME (the #602 finding: NCCL
exchanges IPv6 link-locals with no IPv4 fallback when the interface list is
not restricted). Remote teardown kills workers by /proc-verified PID matched
on their TORCHELASTIC_RUN_ID environment — never an unscoped pattern kill.
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


def remote_worker_pids(host, run_id, *, ssh='ssh', timeout=15):
    """PIDs of remote supervised workers whose environ carries run_id.

    Exact-match discipline: candidates come from the worker module name in
    the command line, then each PID's /proc/<pid>/environ must contain
    TORCHELASTIC_RUN_ID=<run_id> verbatim. A stale or reused PID cannot
    pass the environ check.
    """
    script = (
        "for pid in $(pgrep -f damendaemon.distributed_worker); do"
        " if tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null"
        f" | grep -qx TORCHELASTIC_RUN_ID={run_id}; then echo $pid; fi;"
        " done")
    try:
        result = subprocess.run([ssh, host, script], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(pid) for pid in result.stdout.split() if pid.isdigit()]


def supervised_builder(args, torch):
    """Return runtime_builder(generation) -> (pair, runtime) for the supervisor."""
    worker_host = getattr(args, 'worker_host', None)
    if worker_host:
        # Cross-host generation: one coordinator device, one remote device.
        # Validate the knobs before touching CUDA (CPU-bus tests and honest
        # CLI errors must not depend on device enumeration).
        worker_gpu = getattr(args, 'worker_gpu', None)
        worker_python = getattr(args, 'worker_python', None)
        worker_repo = getattr(args, 'worker_repo', None)
        worker_model = getattr(args, 'worker_model_path', None) or args.path
        rendezvous_host = getattr(args, 'rendezvous_host', None)
        coordinator_ifname = getattr(args, 'coordinator_ifname', None)
        worker_ifname = getattr(args, 'worker_ifname', None)
        if (not worker_gpu or not worker_gpu.startswith('GPU-')
                or not worker_python or not worker_repo or not rendezvous_host
                or rendezvous_host in ('127.0.0.1', 'localhost')
                or not coordinator_ifname or not worker_ifname
                or args.model):
            raise ValueError('cross-host supervised-serve needs --worker-gpu, '
                             '--worker-python, --worker-repo, a LAN '
                             '--rendezvous-host and both NCCL interface names; '
                             'resident additional models are loopback-only '
                             '(the remote worker cannot mirror them)')
        # Explicit coordinator device: required when other devices on this
        # host are busy (enumeration would otherwise pick device 0), and it
        # must bypass gpu_uuids entirely — enumerating here would initialize
        # CUDA against the ambient device list before the builder narrows
        # CUDA_VISIBLE_DEVICES, and torch caches that count (the #637
        # loopback only passed because ambient count == 2 == visible).
        coordinator_gpu = getattr(args, 'coordinator_gpu', None)
        if coordinator_gpu:
            if not coordinator_gpu.startswith('GPU-'):
                raise ValueError('coordinator GPU must be an explicit UUID (GPU-...)')
            uuids = [coordinator_gpu]
        else:
            uuids = gpu_uuids(torch, count=1)
    else:
        uuids = gpu_uuids(torch, count=2)
        # Loopback generation; the cross-host knobs stay unused empty strings
        # so the closure below keeps a single str-typed view of them.
        worker_gpu = worker_python = worker_repo = worker_model = ''
        rendezvous_host = coordinator_ifname = worker_ifname = ''
    additional = [tuple(entry.split('=', 1)) for entry in args.model]
    heartbeat_deadline = getattr(args, 'heartbeat_deadline', 10.0)
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
        store_host = '127.0.0.1' if not worker_host else rendezvous_host
        common = dict(WORLD_SIZE='2', MASTER_ADDR=store_host, MASTER_PORT=str(rendezvous_port),
                      TORCHELASTIC_USE_AGENT_STORE='true', TORCHELASTIC_RUN_ID=run_id,
                      TORCHELASTIC_RESTART_COUNT='0', DAMENDAEMON_DEVICE='cuda',
                      DAMENDAEMON_QUANTIZATION=args.quantization,
                      # Match the coordinator's watchdog configuration (see
                      # distributed_server_cli.py): the abort-based rejoin
                      # needs ASYNC_ERROR_HANDLING off, and a worker whose
                      # coordinator dies exits cleanly at the group timeout
                      # via BLOCKING_WAIT instead of a watchdog abort storm.
                      TORCH_NCCL_ASYNC_ERROR_HANDLING=os.environ.get(
                          'TORCH_NCCL_ASYNC_ERROR_HANDLING', '0'),
                      TORCH_NCCL_BLOCKING_WAIT=os.environ.get(
                          'TORCH_NCCL_BLOCKING_WAIT', '1'))
        store = dist.TCPStore(store_host, rendezvous_port, world_size=None, is_master=True,
                              wait_for_workers=False, timeout=timedelta(seconds=30))
        if worker_host:
            env = dict(common, RANK='1', LOCAL_RANK='0',
                       CUDA_VISIBLE_DEVICES=worker_gpu,
                       NCCL_SOCKET_IFNAME=worker_ifname,
                       NCCL_DEBUG='WARN')
            # The remote worker is detached from the ssh session (nohup +
            # setsid, output to a per-generation remote log): the ssh pipe
            # only carries the spawn — NCCL talks host-to-host directly — so
            # an ssh client death must not kill a serving worker, and the
            # worker's output must not depend on the pipe staying open.
            assignments = ' '.join(f'{key}={value}'
                                   for key, value in sorted(env.items()))
            worker_log = f'/tmp/{run_id}.worker.log'
            # Env assignments must precede nohup/setsid: as a shell env-prefix
            # they are exported into the command's environment; after nohup
            # they would be parsed as arguments to the setsid program.
            remote_command = (f'cd {worker_repo} && PYTHONPATH={worker_repo} '
                              f'{assignments} '
                              f'nohup setsid {worker_python} '
                              f'-m damendaemon.distributed_worker {worker_model}'
                              f' --context {args.context}'
                              f' --quantization {args.quantization}'
                              f' --alias {args.alias}'
                              f' --batch-size {args.batch_size}'
                              f' > {worker_log} 2>&1 < /dev/null &'
                              ' echo WORKER-SPAWNED')
            worker = subprocess.Popen(['ssh', worker_host, remote_command],
                                      start_new_session=True)
        else:
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
        if worker_host:
            parent['NCCL_SOCKET_IFNAME'] = coordinator_ifname
        os.environ.clear(); os.environ.update(parent)
        try:
            pair = StagePair(args.path, args.context, args.quantization, lambda model: [],
                             session=WORKER_SESSION, max_batch=args.batch_size,
                             model_alias=args.alias, additional_models=additional)
            if pair.rank != 0:
                raise ValueError('the supervisor coordinator requires RANK=0')
            runtime = DistributedRuntime(pair, args.alias, args.queue_size,
                                         args.queue_timeout,
                                         client_decode_turns=args.client_decode_turns,
                                         heartbeat_deadline=heartbeat_deadline)
        except BaseException:
            worker.terminate()
            if worker_host:
                # The detached remote worker must not outlive a failed
                # generation build: kill it by its run-id-verified PID.
                for pid in remote_worker_pids(worker_host, run_id):
                    try:
                        subprocess.run(['ssh', worker_host, f'kill {pid}'],
                                        capture_output=True, timeout=10,
                                        check=False)
                    except (OSError, subprocess.SubprocessError):
                        pass
            raise
        pair.supervised_worker = worker
        pair.supervised_store = store
        if worker_host:
            pair.supervised_remote = dict(host=worker_host, run_id=run_id)
        return pair, runtime
    return builder


def teardown_supervised_pair(pair):
    """Generation teardown: kill the worker child, close store and pair."""
    worker = getattr(pair, 'supervised_worker', None)
    remote = getattr(pair, 'supervised_remote', None)
    if remote is not None:
        for pid in remote_worker_pids(remote['host'], remote['run_id']):
            try:
                subprocess.run(['ssh', remote['host'], f'kill {pid}'],
                               capture_output=True, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
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

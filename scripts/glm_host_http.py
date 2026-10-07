"""Owned loopback HTTP lifetime for an explicit native GLM/Flash/DeepSeek host topology."""
import signal
import threading
import time
from dllm.reference_serving import handler_for_reference
from dllm.runtime import BusyError
from dllm.server import BoundedHTTPServer
from scripts.run_cpu_host_reference import emit


def serve(worker,text,args,auth,*,family='glm'):
    if family=='deepseek':
        from dllm.deepseek_host_generation import HostDeepSeekGenerator as Runner
        from dllm.deepseek_host_serving import HostDeepSeekReferenceRuntime as Runtime
        from dllm.queued_deepseek_host_serving import QueuedHostDeepSeekReferenceRuntime as QueuedRuntime
    elif family=='flash':
        from dllm.glm_flash_host_generation import HostFlashGenerator as Runner
        from dllm.glm_flash_host_serving import HostFlashReferenceRuntime as Runtime
        from dllm.queued_glm_flash_host_serving import QueuedHostFlashReferenceRuntime as QueuedRuntime
    elif family=='glm':
        from dllm.glm_host_generation import HostGlmGenerator as Runner
        from dllm.glm_host_serving import HostGlmReferenceRuntime as Runtime
        from dllm.queued_glm_host_serving import QueuedHostGlmReferenceRuntime as QueuedRuntime
    else:raise ValueError('unsupported native host HTTP family')
    runner=Runner(worker,text,chunk_size=args.prefill_chunk)
    runtime=(QueuedRuntime(runner,args.alias,queue_size=args.queue_size,queue_timeout=args.queue_timeout)
             if args.queue_size else Runtime(runner,args.alias))
    server=None;previous={};primary=None;stopping=threading.Event()
    graceful=False
    def interrupt(*_):stopping.set();runtime.stopping.set()
    try:
        server=BoundedHTTPServer(('127.0.0.1',args.http_port),handler_for_reference(runtime,auth))
        server.timeout=min(.2,args.heartbeat_interval/2)
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,interrupt)
        emit(dict(event='http_ready',url=f'http://127.0.0.1:{server.server_port}',model=args.alias,
            model_type=runtime.family,scope='cpu_reference',native_chat=True,dtype='float32',
            official_model_qualified=False,queue_enabled=bool(args.queue_size),queue_capacity=args.queue_size,**runtime.execution_status()))
        next_ping=time.monotonic()+args.heartbeat_interval
        while not stopping.is_set():
            if args.stop_after_drain and runtime.scheduler_status()['drained']:
                stopping.set();runtime.stopping.set();break
            if not runtime.is_healthy():
                if stopping.is_set():break
                raise RuntimeError('native host HTTP epoch ended')
            server.handle_request()
            if (not stopping.is_set() and time.monotonic()>=next_ping
                    and worker.lock.acquire(blocking=False)):
                try:
                    if stopping.is_set():continue
                    if not runtime.is_healthy():raise RuntimeError('native host HTTP epoch ended')
                    # PING also keeps backpressured streams alive. It leaves
                    # slot state unchanged and shares the worker
                    # command lock. A racing request waits there, not a false
                    # HTTP busy rejection from a maintenance admission lock.
                    worker.ping()
                    next_ping=time.monotonic()+args.heartbeat_interval
                finally:worker.lock.release()
        # STOP is safe only while no HTTP iterator owns generation. Otherwise
        # interrupt the epoch; never label active cancellation graceful.
        queue_state=runtime.scheduler_status() if args.queue_size else {}
        queued=queue_state.get('worker_owned_generation_requests',0)+queue_state.get('queued_generation_requests',0)
        if not queued and runtime.lock.acquire(blocking=False):
            try:
                if not runtime._terminal():worker.stop();graceful=True
            finally:runtime.lock.release()
        if not graceful:raise KeyboardInterrupt
    except BaseException as error:primary=error;raise
    finally:
        runtime.stopping.set();failures=[]
        # Interrupt receives before waiting for non-daemon HTTP request threads.
        try:runtime.close()
        except BusyError:pass
        except BaseException as error:failures.append(error)
        if server is not None:
            try:server.server_close()
            except BaseException as error:failures.append(error)
        try:runtime.close()
        except BaseException as error:failures.append(error)
        for sig,handler in previous.items():signal.signal(sig,handler)
        if failures:
            if primary is not None:primary.add_note('native host HTTP cleanup failed; retain owners for retry')
            else:raise failures[0]
    emit(dict(event='stopped',scope='cpu_reference',execution='cpu_host_topology',graceful_stop=True))

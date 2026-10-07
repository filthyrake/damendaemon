"""Owned loopback HTTP lifetime for an explicit native GLM host topology."""
import signal
import threading
import time
from dllm.glm_host_generation import HostGlmGenerator
from dllm.glm_host_serving import HostGlmReferenceRuntime
from dllm.reference_serving import handler_for_reference
from dllm.runtime import BusyError
from dllm.server import BoundedHTTPServer
from scripts.run_cpu_host_reference import emit


def serve(worker,text,args,auth):
    runner=HostGlmGenerator(worker,text,chunk_size=args.prefill_chunk)
    runtime=HostGlmReferenceRuntime(runner,args.alias)
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
            official_model_qualified=False,queue_enabled=False,**runtime.execution_status()))
        next_ping=time.monotonic()+args.heartbeat_interval
        while not stopping.is_set():
            if not runtime.is_healthy():
                if stopping.is_set():break
                raise RuntimeError('native GLM host HTTP epoch ended')
            server.handle_request()
            if (not stopping.is_set() and not runtime.active and time.monotonic()>=next_ping
                    and worker.lock.acquire(blocking=False)):
                try:
                    if stopping.is_set():continue
                    if not runtime.is_healthy():raise RuntimeError('native GLM host HTTP epoch ended')
                    # PING leaves slot state unchanged and shares the worker
                    # command lock. A racing request waits there, not a false
                    # HTTP busy rejection from a maintenance admission lock.
                    worker.ping()
                    next_ping=time.monotonic()+args.heartbeat_interval
                finally:worker.lock.release()
        # STOP is safe only while no HTTP iterator owns generation. Otherwise
        # interrupt the epoch; never label active cancellation graceful.
        if runtime.lock.acquire(blocking=False):
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
            if primary is not None:primary.add_note('native GLM host HTTP cleanup failed; retain owners for retry')
            else:raise failures[0]
    emit(dict(event='stopped',scope='cpu_reference',execution='cpu_host_topology',graceful_stop=True))

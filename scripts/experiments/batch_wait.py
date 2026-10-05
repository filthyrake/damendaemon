"""Explicit idle-group coalescing experiment; preserves FIFO management barriers."""
import math
import os
import time


def wait_for_batch(runtime,first,delay):
    if runtime.batch_size<=1:return
    deadline=time.monotonic()+delay
    with runtime.condition:
        while not runtime.closed and not first.cancelled.is_set():
            count=0
            for request in runtime.pending:
                if request.done.is_set() or request.cancelled.is_set():continue
                # Do not wait across management or another model's FIFO barrier.
                if request.command is not None or request.alias!=first.alias:return
                count+=1
                if count>=runtime.batch_size-1:return
            remaining=deadline-time.monotonic()
            if remaining<=0:return
            runtime.condition.wait(timeout=remaining)


def install_validation_dispatch():
    from dllm.scheduler import BatchedRuntime
    milliseconds=float(os.environ['DLLM_EXPERIMENTAL_BATCH_WAIT_MS'])
    if not math.isfinite(milliseconds) or not 0<milliseconds<=10:
        raise ValueError('experimental batch wait must be >0 and <=10 milliseconds')
    original=BatchedRuntime._group
    def group(runtime,first,executor,tier,activation):
        wait_for_batch(runtime,first,milliseconds/1000)
        return original(runtime,first,executor,tier,activation)
    BatchedRuntime._group=group
    return dict(name='experimental idle-group coalescing',milliseconds=milliseconds,production_dispatch=False)

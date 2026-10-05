"""Bounded idle-group admission; never waits inside active decode iterations."""
import time


def wait_for_batch(runtime,first,delay):
    if delay<=0 or runtime.batch_size<=1:return
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


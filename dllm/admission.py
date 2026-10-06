"""Bounded idle-group admission; never waits inside active decode iterations."""
import time
import threading


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



def validate_queue_limits(capacity, timeout):
    """Reject invalid semaphore counts and waits before runtime side effects."""
    if type(capacity) is not int or capacity <= 0:
        raise ValueError('queue size must be a positive integer')
    if (type(timeout) not in (int, float)
            or not 0 < timeout <= threading.TIMEOUT_MAX):
        raise ValueError('queue timeout must be finite, positive and within the platform wait limit')

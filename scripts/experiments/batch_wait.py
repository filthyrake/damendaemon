"""Explicit idle-group coalescing experiment; preserves FIFO management barriers."""
import math
import os


from damendaemon.admission import wait_for_batch

def install_validation_dispatch():
    from damendaemon.scheduler import BatchedRuntime
    milliseconds=float(os.environ['DAMENDAEMON_EXPERIMENTAL_BATCH_WAIT_MS'])
    if not math.isfinite(milliseconds) or not 0<milliseconds<=10:
        raise ValueError('experimental batch wait must be >0 and <=10 milliseconds')
    original=BatchedRuntime._group
    def group(runtime,first,executor,tier,activation):
        wait_for_batch(runtime,first,milliseconds/1000)
        return original(runtime,first,executor,tier,activation)
    BatchedRuntime._group=group
    return dict(name='experimental idle-group coalescing',milliseconds=milliseconds,production_dispatch=False)

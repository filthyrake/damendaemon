"""Explicit single-owner native reference HTTP over a prepared CPU host session."""
from contextlib import closing
import threading
from .reference_serving import ReferenceRuntime
from .runtime import BusyError


class NativeHostReferenceRuntime(ReferenceRuntime):
    """Own one native host epoch; failed or abandoned active requests are terminal.

    Peers must keep stepping externally. Readiness describes observed transport
    state, not a fresh network probe. Use close() to interrupt host I/O; retain
    and retry this owner when cleanup fails or an iterator is still active.
    """
    runner_type=None
    model_type=None
    supports_images=False

    def __init__(self,runner,alias='model'):
        if self.runner_type is None or type(runner) is not self.runner_type:raise ValueError('host HTTP requires an explicit native reference host runner')
        if not isinstance(alias,str) or not 1<=len(alias)<=128:raise ValueError('invalid reference host alias')
        if not runner._active.acquire(blocking=False):raise ValueError('host runner is active')
        try:
            worker=runner.worker
            if (getattr(runner,'_http_owner',None) is not None or runner.completion.stopped.is_set()
                    or worker.closed or worker.failed or not worker.ready or runner.failed_cleanups
                    or any(worker.channel.state.lengths)
                    or any(channel.transport.closed or channel.transport.failed for channel in worker.channels)):
                raise ValueError('host HTTP requires an unclaimed healthy idle native runner')
            self.runner,self.alias=runner,alias
            self.lock=threading.Lock();self.stopping=threading.Event()
            self.images=self.supports_images;self.family=self.model_type;self.context=runner.max_context
            self.active=self.closed=False;self.owner=object()
            self.stage_layers=tuple(worker.topology.ranges)
            self.source_model_id=worker.topology.source_identity.hex()
            self.topology_identity=worker.topology.identity.hex()
            self.model_identity_verified=getattr(worker,'model_identity_verified',False)
            runner._http_owner=self.owner
        finally:runner._active.release()

    def _terminal(self):
        runner=self.runner
        if runner is None:return True
        worker=runner.worker
        return (runner.completion.stopped.is_set() or worker.closed or worker.failed or not worker.ready
                or any(channel.transport.closed or channel.transport.failed for channel in worker.channels))

    def execution_status(self):
        return dict(execution='cpu_host_topology',distributed=True,stage_layers=[list(pair) for pair in self.stage_layers],
                    source_model_id=self.source_model_id,topology_identity=self.topology_identity,
                    model_identity_verified=self.model_identity_verified,epoch_terminal=self._terminal(),
                    **(dict(vision_identity_verified=self.model_identity_verified) if self.images else {}))

    def cleanup_pending(self):
        runner=self.runner
        return bool(runner is not None and (runner.failed_cleanups
            or ((self.stopping.is_set() or self._terminal()) and not runner.worker.closed)))

    def is_healthy(self):
        return not self.stopping.is_set() and not self._terminal() and not self.cleanup_pending()

    def _cleanup(self):
        runner=self.runner
        if runner is None:return
        if self._terminal():self.stopping.set()
        try:
            if self.stopping.is_set():
                runner.close();self.runner=None;self.closed=True
            else:runner.cleanup()
        except BaseException:
            self.stopping.set()
            raise

    def close(self):
        self.stopping.set()
        if not self.lock.acquire(blocking=False):
            # Closing the transport interrupts an active receive before waiting
            # for its model lock. The iterator retains HTTP ownership meanwhile.
            runner=self.runner
            if runner is not None:runner.close()
            raise BusyError('reference host shutdown is waiting for its active iterator')
        try:self._cleanup()
        finally:self.lock.release()

    def generate(self,*args,**options):
        if not self.is_healthy():raise BusyError('reference host epoch is unavailable; prepare a new session')
        with closing(super().generate(*args,**options)) as events:
            for event in events:
                if self.stopping.is_set() or self._terminal():raise BusyError('reference host epoch ended during generation')
                yield event

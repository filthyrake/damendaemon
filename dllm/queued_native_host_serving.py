"""Bounded serialized native reference host admission with terminal-epoch ownership."""
from .native_host_serving import NativeHostReferenceRuntime
from .queued_reference_serving import QueuedReferenceRuntime
from .runtime import BusyError


class QueuedNativeHostReferenceRuntime(QueuedReferenceRuntime,NativeHostReferenceRuntime):
    """Reuse queue bounds/scopes; only a healthy undispatched epoch is reusable.

    The cooperative initializer reaches the explicit host runtime, not the
    generic local-model admission. Queued cancellation leaves the session alone;
    cancellation after dispatch retires it and flushes every waiting request.
    """
    def _admit(self):
        super()._admit()
        if self._terminal():raise BusyError('reference host epoch is unavailable; prepare a new session')

    def _generate_owned(self,request):
        try:
            yield from NativeHostReferenceRuntime.generate(self,self.alias,request.messages,request.maximum,**request.options)
        except BaseException as error:
            if self._terminal():
                if self.failure is None:self.failure=error
                self.stopping.set()
            raise

    def close(self):
        with self.admission_lock:self.stopping.set()
        primary=None;runner=self.runner
        # The queue thread may be waiting for a remote frame. Interrupt that
        # receive before the inherited bounded join, preserving retry ownership.
        if runner is not None:
            try:runner.close()
            except BaseException as error:primary=error
        try:super().close()
        except BaseException as error:
            if primary is None:primary=error
            else:primary.add_note('queued host HTTP shutdown also failed; retain runtime and retry close')
        if primary is not None:raise primary

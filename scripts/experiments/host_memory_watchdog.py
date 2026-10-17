"""Host-side protection for an owned experiment, independent of GPU accounting."""
import threading

from damendaemon.rocm_policy import host_available


class HostMemoryWatchdog:
    def __init__(self, expire, report, reserve_bytes=20*1024**3, read=host_available):
        self.expire, self.report, self.read = expire, report, read
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        report.update(reserve_bytes=reserve_bytes, minimum_available_bytes=None,
                      last_available_bytes=None, triggered=False, error=None)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        self.thread.join()

    def _watch(self):
        while not self.stop.is_set():
            try:
                available = self.read()
                self.report['last_available_bytes'] = available
                previous = self.report['minimum_available_bytes']
                self.report['minimum_available_bytes'] = available if previous is None else min(previous, available)
                if available > self.report['reserve_bytes']:
                    if self.stop.wait(.1):
                        return
                    continue
            except Exception as error:
                self.report['error'] = repr(error)
            self.report['triggered'] = True
            self.expire()
            return

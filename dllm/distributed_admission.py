"""Fair removal from one bounded resident-model queue, with no extra pool."""
import queue
import time
from .fair_admission import FairAdmission


class DistributedAdmission(queue.Queue):
    """Queue's mutex serializes selection with puts/gets and capacity accounting.

    There is one resident model and no management barriers in this queue. Opaque
    scopes are captured by Request on the HTTP thread; they are not bearer keys.
    Cancelled/expired entries return to the owner's existing rejection path before
    choosing a live turn, without charging their lane or adding a pending pool.
    """
    def __init__(self,maxsize,timeout):
        super().__init__(maxsize)
        self.timeout=timeout;self.fair=FairAdmission()

    def _get(self):
        now=time.monotonic()
        stale=next((request for request in self.queue if request.cancelled.is_set()
                    or now-request.submitted>=self.timeout),None)
        chosen=stale if stale is not None else self.fair.choose('resident',self.queue)
        for index,request in enumerate(self.queue):
            if request is chosen:
                del self.queue[index]
                break
        if not self.queue:self.fair.reset()
        return chosen

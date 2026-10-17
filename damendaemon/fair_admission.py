"""Bounded identity-based turns within one generation model group."""


class FairAdmission:
    def __init__(self):
        self.reset()

    def reset(self):
        self.alias=None
        self.turn=0
        self.recent={}

    def choose(self,alias,requests):
        """Choose each lane's oldest request, favoring least recently served.

        Caller supplies only live requests before the next model/command barrier.
        Scope objects are opaque identities, never bearer strings or body fields.
        History holds at most one entry per currently waiting lane; references
        prevent identity reuse while a lane's last turn remains recorded.
        """
        if not requests:raise ValueError('fair admission requires waiting requests')
        if alias!=self.alias:
            self.reset();self.alias=alias
        oldest={}
        for request in requests:oldest.setdefault(id(request.cache_scope),request)
        self.recent={key:value for key,value in self.recent.items() if key in oldest}
        chosen=min(oldest.values(),key=lambda r:self.recent.get(id(r.cache_scope),(None,-1))[1])
        self.turn+=1
        self.recent[id(chosen.cache_scope)]=(chosen.cache_scope,self.turn)
        return chosen

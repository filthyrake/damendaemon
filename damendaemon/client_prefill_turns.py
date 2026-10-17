"""Weighted prompt-token service across live opaque clients, with rotation."""
from .client_decode_turns import SERVICE_UNITS


class ClientPrefillTurns:
    def __init__(self):self.reset()

    def reset(self):
        self.service={};self.recent={};self.turn=0

    def choose(self,ready):
        """Choose and charge one (key, request, positive chunk-token count).

        The owner supplies only runnable pending chunks and serializes calls.
        Completed/cancelled/absent lanes lose history; a new lane starts at the
        current floor. Normalized service skew stays within one maximum chunk.
        Charging dispatch attempts is conservative if the backend then fails.
        Immutable weights 1..16 divide actual token cost in exact integer units.
        No time estimate, slot reservation or future token debt is used.
        """
        if not ready:self.reset();return None
        if any(type(cost) is not int or not 1<=cost<=2**20 for _,_,cost in ready):
            raise ValueError('client prefill turns require bounded positive chunk tokens')
        lanes={}
        for item in ready:lanes.setdefault(id(item[1].cache_scope),[]).append(item)
        costs={}
        for key,lane in lanes.items():
            weight=getattr(lane[0][1].cache_scope,'prefill_weight',1)
            if type(weight) is not int or not 1<=weight<=16:raise ValueError('invalid ready-client prefill weight')
            costs[key]=SERVICE_UNITS//weight
        live={id(request) for _,request,_ in ready}
        self.recent={key:value for key,value in self.recent.items() if key in live}
        retained={key:value for key,value in self.service.items() if key in lanes}
        floor=min((value[1] for value in retained.values()),default=0)
        service={key:(lane[0][1].cache_scope,*retained.get(key,(None,floor,-1))[1:]) for key,lane in lanes.items()}
        key=min(lanes,key=lambda key:service[key][1:])
        item=min(lanes[key],key=lambda item:self.recent.get(id(item[1]),(None,-1))[1])
        _,request,cost=item;self.turn+=1
        self.recent[id(request)]=(request,self.turn)
        service[key]=(request.cache_scope,service[key][1]+cost*costs[key],self.turn)
        floor=min(value[1] for value in service.values())
        self.service={key:(scope,used-floor,turn) for key,(scope,used,turn) in service.items()}
        return item

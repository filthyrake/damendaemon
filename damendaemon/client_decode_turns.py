"""Weighted ready-client token rows with bounded fractional credit and rotation."""
import math

SERVICE_UNITS=math.lcm(*range(1,17))


class ClientDecodeTurns:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn=0
        self.recent={}
        self.credits={}

    def choose(self,ready):
        """Advance a common service interval across ready opaque client lanes.

        Items are (slot, request, prompt_length). Identity, not equality, groups
        scopes; None is the shared anonymous lane. Only ready requests remain in
        history, and the retained object prevents ID reuse. No token debt accrues
        while an entire client lane is backpressured or absent. Weights 1..16
        set token-row shares; defaults preserve the largest equal share. Each
        credit is less than one row cost, using exact bounded integer units.
        Caller supplies at most the scheduler's active batch and serializes access on its worker thread.
        """
        lanes={}
        for item in ready:lanes.setdefault(id(item[1].cache_scope),[]).append(item)
        live={id(item[1]) for item in ready}
        self.recent={key:value for key,value in self.recent.items() if key in live}
        if not ready:
            self.reset();return []
        costs={};balances={}
        for key,lane in lanes.items():
            scope=lane[0][1].cache_scope;weight=getattr(scope,'decode_weight',1)
            if type(weight) is not int or not 1<=weight<=16:raise ValueError('invalid ready-client decode weight')
            costs[key]=SERVICE_UNITS//weight
            balances[key]=self.credits.get(key,(None,0))[1]
        if all(cost==SERVICE_UNITS for cost in costs.values()):balances=dict.fromkeys(costs,0)
        # The first lane to exhaust its ready rows bounds a common interval.
        # Fractional credits let small batches honor ratios across iterations.
        interval=min(len(lanes[key])*cost-balances[key] for key,cost in costs.items())
        chosen=set();credits={}
        for key,lane in lanes.items():
            share,remainder=divmod(balances[key]+interval,costs[key])
            credits[key]=(lane[0][1].cache_scope,remainder)
            lane.sort(key=lambda item:self.recent.get(id(item[1]),(None,-1))[1])
            for _,request,_ in lane[:share]:
                chosen.add(id(request));self.turn+=1
                self.recent[id(request)]=(request,self.turn)
        self.credits=credits
        return [item for item in ready if id(item[1]) in chosen]

"""Equal decode rows per ready client, with bounded within-client rotation."""


class ClientDecodeTurns:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn=0
        self.recent={}

    def choose(self,ready):
        """Select a largest equal share from each ready opaque client lane.

        Items are (slot, request, prompt_length). Identity, not equality, groups
        scopes; None is the shared anonymous lane. Only ready requests remain in
        history, and the retained object prevents ID reuse. No token debt accrues
        while a request is backpressured or absent. Caller supplies at most the
        scheduler's active batch and serializes access on its worker thread.
        """
        lanes={}
        for item in ready:lanes.setdefault(id(item[1].cache_scope),[]).append(item)
        live={id(item[1]) for item in ready}
        self.recent={key:value for key,value in self.recent.items() if key in live}
        if not ready:
            self.reset();return []
        share=min(map(len,lanes.values()))
        chosen=set()
        for lane in lanes.values():
            lane.sort(key=lambda item:self.recent.get(id(item[1]),(None,-1))[1])
            chosen.update(id(item[1]) for item in lane[:share])
        self.turn+=1
        selected=[item for item in ready if id(item[1]) in chosen]
        for _,request,_ in selected:self.recent[id(request)]=(request,self.turn)
        return selected

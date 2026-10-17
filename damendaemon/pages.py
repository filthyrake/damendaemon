"""Host ownership of a bounded KV page pool; no tensor or GPU dependencies."""


class PageCapacityError(Exception):
    pass


class PagePool:
    def __init__(self, tokens, slots, context, page_size=16):
        if (any(type(x) is not int or x <= 0 for x in (tokens, slots, context, page_size))
                or tokens % page_size or tokens < slots * page_size):
            raise ValueError('KV token budget must be a page multiple and fit one page per slot')
        self.page_size, self.context = page_size, context
        self.capacity = tokens // page_size
        self.pages = [[] for _ in range(slots)]
        self.free = list(reversed(range(self.capacity)))
        self.revision = 0
        self.high_water_tokens = 0

    def needed(self, tokens):
        if type(tokens) is not int or not 0 <= tokens <= self.context:
            raise ValueError('invalid KV reservation length')
        return (tokens + self.page_size - 1) // self.page_size

    def resize_many(self, requests, replace=False):
        """Validate the whole transaction before changing any slot's ownership."""
        if len({slot for slot, _ in requests}) != len(requests):
            raise ValueError('duplicate KV slot')
        sizes = {}
        for slot, tokens in requests:
            if type(slot) is not int or not 0 <= slot < len(self.pages):
                raise ValueError('invalid KV slot')
            count = self.needed(tokens)
            sizes[slot] = count if replace else max(count, len(self.pages[slot]))
        delta = sum(count - len(self.pages[slot]) for slot, count in sizes.items())
        if delta > len(self.free):
            raise PageCapacityError('KV page pool is full')
        for slot, count in sizes.items():
            if count < len(self.pages[slot]):
                self.free.extend(self.pages[slot][count:])
                del self.pages[slot][count:]
                self.revision += 1
        for slot, count in sizes.items():
            missing = count - len(self.pages[slot])
            if missing:
                self.pages[slot].extend(self.free.pop() for _ in range(missing))
                self.revision += 1

        self.high_water_tokens = max(self.high_water_tokens, self.used_tokens)

    def release(self, slot):
        self.resize_many([(slot, 0)], replace=True)

    @property
    def used_tokens(self):
        return (self.capacity - len(self.free)) * self.page_size

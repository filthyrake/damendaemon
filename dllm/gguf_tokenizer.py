"""GGUF global special-piece partitioning shared across tokenizer families."""
from bisect import bisect_left
import json
from types import SimpleNamespace


def validate_special_overlaps(tokens, types):
    """Equal-byte-length overlapping specials have no portable reference order.

    llama.cpp sorts special pieces by byte length without a tie breaker. Only
    distinct pieces with a proper suffix/prefix overlap can depend on that tie.
    Binary search avoids comparing every pair in large same-length groups.
    """
    groups = {}
    for text, kind in zip(tokens, types):
        if kind in (2, 3, 4):
            raw = text.encode('utf-8')
            groups.setdefault(len(raw), []).append(raw)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort()
        initials = {piece[0] for piece in group}
        for piece in group:
            for offset in range(1, len(piece)):
                if piece[offset] not in initials:
                    continue
                suffix = piece[offset:]
                start = bisect_left(group, suffix)
                # The first matching piece may be itself (self-overlap is safe).
                for other in group[start:start + 2]:
                    if other != piece and other.startswith(suffix):
                        raise ValueError('GGUF has equal-length overlapping special tokens')


class SpecialTokenTokenizer:
    """Partition longer special pieces before normal-fragment tokenization."""
    def __init__(self, backend):
        self.backend_tokenizer = backend
        self._pieces = {token.content: index for index, token in backend.get_added_tokens_decoder().items()}
        self._special_order = sorted(self._pieces, key=lambda text: len(text.encode('utf-8')), reverse=True)
        policy = json.loads(backend.to_str()).get('post_processor')
        self._prefix, self._suffix = [], []
        if policy:
            target = self._prefix
            for entry in policy['single']:
                if 'Sequence' in entry:
                    target = self._suffix
                else:
                    target.extend(policy['special_tokens'][entry['SpecialToken']['id']]['ids'])

    def encode(self, text, add_special_tokens=True):
        if not isinstance(text, str):
            raise TypeError('GGUF tokenizer input must be text')
        ids = list(self._prefix) if add_special_tokens else []
        # Match GGUF's global longest-special-first partitioning, including a
        # longer special beginning inside a shorter one. Leftmost regex matching
        # would incorrectly consume the shorter special in that case.
        fragments = [text]
        for piece in self._special_order:
            divided = []
            for fragment in fragments:
                if isinstance(fragment, int):
                    divided.append(fragment)
                    continue
                parts = fragment.split(piece)
                for index, part in enumerate(parts):
                    if index:
                        divided.append(self._pieces[piece])
                    if part:
                        divided.append(part)
            fragments = divided
        for fragment in fragments:
            if isinstance(fragment, int):
                ids.append(fragment)
                continue
            if not fragment:
                continue
            ids.extend(self._encode_fragment(fragment))
        if add_special_tokens:
            ids.extend(self._suffix)
        return SimpleNamespace(ids=ids)

    def encode_batch(self, sequences, add_special_tokens=True):
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in sequences]

    def _encode_fragment(self, fragment):
        return self.backend_tokenizer.encode(fragment, add_special_tokens=False).ids

    def __getattr__(self, name):
        return getattr(self.backend_tokenizer, name)

    def to_str(self, pretty=False):
        return json.dumps(dict(dllm_gguf_special=1, backend=self.backend_tokenizer.to_str()),
                          indent=2 if pretty else None)

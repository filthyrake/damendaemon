"""Incremental CPU tokenizer ByteLevel decoding without whole-history copies."""
import codecs
from tokenizers import decoders


def _byte_alphabet():
    visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
    result={chr(value):value for value in visible};extra=256
    for value in range(256):
        if value not in visible:result[chr(extra)]=value;extra+=1
    return result


BYTE_VALUES=_byte_alphabet()


class ByteLevelStream:
    def __init__(self,tokenizer,allowed_added):
        if not isinstance(tokenizer.decoder,decoders.ByteLevel):
            raise ValueError('incremental output requires a ByteLevel tokenizer')
        self.tokenizer=tokenizer;self.added=tokenizer.get_added_tokens_decoder()
        self.allowed=frozenset(allowed_added)
        self.utf8=codecs.getincrementaldecoder('utf-8')(errors='replace')

    def feed(self,token):
        if token in self.added:
            value=self.added[token].content
            if value not in self.allowed:raise ValueError('unsupported generated added/control token')
            decoded=self.utf8.decode(b'',final=True)+value
            self.utf8.reset()
            return decoded
        value=self.tokenizer.id_to_token(token)
        try:raw=bytes(BYTE_VALUES[char] for char in value)
        except (KeyError,TypeError) as error:raise ValueError('invalid ByteLevel vocabulary token') from error
        return self.utf8.decode(raw,final=False)

    def finish(self):return self.utf8.decode(b'',final=True)

"""Text stopping with safe incremental output across token boundaries."""


def stop_sequences(value):
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if (not isinstance(values, list) or not 1 <= len(values) <= 4
            or any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in values)):
        raise ValueError('stop must contain one to four nonempty strings of at most 256 characters')
    return tuple(values)


def text_delta(decoded, emitted, stops=(), final=False):
    """Return only irrevocable text, retaining incomplete stop/UTF-8 suffixes."""
    matches = [index for text in stops if (index := decoded.find(text)) >= 0]
    stopped = bool(matches)
    if stopped:
        visible = decoded[:min(matches)]
    elif decoded.endswith('\ufffd') and not final:
        return '', emitted, False
    else:
        held = 0
        if not final:
            for text in stops:
                for count in range(min(len(text)-1, len(decoded)), held, -1):
                    if decoded.endswith(text[:count]):
                        held = count
                        break
        visible = decoded[:-held] if held else decoded
    if not visible.startswith(emitted):
        raise RuntimeError('tokenizer revised already streamed text')
    return visible[len(emitted):], visible, stopped

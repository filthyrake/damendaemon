"""Atomic JSON evidence checkpoints; no hardware or runtime initialization."""
import json
from pathlib import Path
import tempfile


MAX_REPORT_BYTES = 64 * 1024 * 1024


def load_report(path, *, max_bytes=MAX_REPORT_BYTES):
    """Read a bounded, unambiguous UTF-8 object; numerical policy is schema-owned."""
    if type(max_bytes) is not int or max_bytes<1:raise ValueError('invalid report byte limit')
    with Path(path).open('rb') as reader:
        raw=reader.read(max_bytes+1)
    if len(raw)>max_bytes:raise ValueError('report exceeds byte limit')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate report field')
            result[key]=value
        return result
    try:
        report=json.loads(raw.decode('utf-8'),object_pairs_hook=unique)
    except RecursionError as error:
        raise ValueError('report nesting is too deep') from error
    if not isinstance(report,dict):raise ValueError('report must be an object')
    return report


def save_report(path, report):
    """Replace only after a complete write; keep prior evidence on failure.

    Unique same-directory temporaries also permit nested signal checkpoints.
    This provides atomic replacement, not fsync/power-loss durability.
    """
    path = Path(path)
    temporary = None
    try:
        # A signal handler can checkpoint while the main thread is writing.
        # Unique temporaries prevent its report from sharing an open inode.
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch: damendaemon is being ported to assembly module by
module; this module's entire logic lives in native/stopping_asm.s.
The bridge contains no decision logic — it compiles and caches the
object, resolves the runtime ABI (libm + CPython C API addresses,
which eager flat-namespace binds cannot reach from a dlopen'd image),
installs it via _dd_init, and converts str <-> UTF-8 bytes at the
boundary (the asm works on bytes).

Ported behavior notes (see PORTING_STATUS.md):
- scans are byte-wise UTF-8 rather than code-point-wise; boundary
  round-trips use surrogatepass/surrogateescape so inputs the original
  accepted still round-trip
- stop_sequences uses exact types (str/list), matching the original's
  isinstance checks for exact types
"""
import ctypes
import os
import subprocess
import sys

_REVISION = 'tokenizer revised already streamed text'


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'stopping_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libstopping_asm.dylib')
    os.makedirs(cache, exist_ok=True)
    if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(source):
        subprocess.run([os.environ.get('CC', 'clang'), '-arch', 'arm64', '-shared', '-O2',
                        '-mmacosx-version-min=11.0', '-undefined', 'dynamic_lookup',
                        '-o', target, source], check=True)
    return target


def _address(dll, name):
    return ctypes.cast(getattr(dll, name), ctypes.c_void_p).value


_lib = ctypes.PyDLL(_build())  # PyDLL: keep the GIL — the asm calls back into CPython
# macOS dlsym expects C names (no asm-level leading underscore)
_lib.dd_init.argtypes = [ctypes.c_void_p]
_lib.dd_text_delta.restype = ctypes.c_int
_lib.dd_text_delta.argtypes = [ctypes.py_object, ctypes.py_object,
                               ctypes.py_object, ctypes.c_int, ctypes.c_void_p]
_lib.dd_stop_sequences.restype = ctypes.py_object
_lib.dd_stop_sequences.argtypes = [ctypes.py_object]

_libc = ctypes.CDLL('libSystem.dylib')
_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/stopping_asm.s (offsets in bytes):
#   0 memmem  8 memcmp  16 PyTuple_New  24 PyUnicode_GetLength
#   32 PyUnicode_AsUTF8String  40 Py_DecRef  48 PyErr_Clear
#   56 PyErr_SetString  64 ValueError  72 None  80 str  88 list
_iface = (ctypes.c_void_p * 12)(
    _address(_libc, 'memmem'), _address(_libc, 'memcmp'),
    _address(_python, 'PyTuple_New'), _address(_python, 'PyUnicode_GetLength'),
    _address(_python, 'PyUnicode_AsUTF8String'), _address(_python, 'Py_DecRef'),
    _address(_python, 'PyErr_Clear'), _address(_python, 'PyErr_SetString'),
    id(ValueError), id(None), id(str), id(list),
)
_lib.dd_init(ctypes.byref(_iface))


def stop_sequences(value):
    """Validate and normalize stop strings — logic in assembly."""
    return _lib.dd_stop_sequences(value)


def text_delta(decoded, emitted, stops=(), final=False):
    """Return only irrevocable text — logic in assembly."""
    data = decoded.encode('utf-8', 'surrogatepass')
    prior = emitted.encode('utf-8', 'surrogatepass')
    needles = tuple(stop.encode('utf-8', 'surrogatepass') for stop in stops)
    out = (ctypes.c_int64 * 5)()
    status = _lib.dd_text_delta(data, prior, needles, 1 if final else 0, ctypes.byref(out))
    if status != 0:
        raise RuntimeError(_REVISION)
    delta = ctypes.string_at(out[0], out[1]).decode('utf-8', 'surrogateescape')
    visible = ctypes.string_at(out[2], out[3]).decode('utf-8', 'surrogateescape')
    return delta, visible, bool(out[4])

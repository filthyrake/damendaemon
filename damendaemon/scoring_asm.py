"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 12: options logic lives in
native/scoring_asm.s (validate_raw delegation with keyword args, the
isinstance/count/type/negativity/distinctness candidate gates, and the
(prompt, list(candidates)) copy construction — all via the C API so
exception semantics and list-copy identity are preserved); this bridge
only compiles/caches the dylib, resolves the interface table, and
binds the entry point.
"""
import ctypes
import os
import subprocess
import sys

from .prompts import validate_raw

_HOLD_VALIDATE_RAW = validate_raw  # keep the iface entry alive


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'scoring_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libscoring_asm.dylib')
    os.makedirs(cache, exist_ok=True)
    if not os.path.exists(target) or os.path.getmtime(target) < os.path.getmtime(source):
        subprocess.run([os.environ.get('CC', 'clang'), '-arch', 'arm64', '-shared', '-O2',
                        '-mmacosx-version-min=11.0', '-undefined', 'dynamic_lookup',
                        '-o', target, source], check=True)
    return target


def _address(dll, name):
    return ctypes.cast(getattr(dll, name), ctypes.c_void_p).value


_lib = ctypes.PyDLL(_build())
_lib.dd_init.argtypes = [ctypes.c_void_p]
_lib.dd_options.restype = ctypes.py_object
_lib.dd_options.argtypes = [ctypes.py_object] * 3

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/scoring_asm.s (offsets/8):
#  0 (unused GetAttrString slot)  1 PyObject_Call  2 PyObject_IsInstance
#  3 PyList_Type  4 PyLong_Type  5 PyObject_Size  6 PyObject_RichCompareBool
#  7 PySet_New  8 Py_DecRef  9 PyTuple_New  10 PyTuple_SetItem
# 11 PyDict_New  12 PyDict_SetItem  13 PyUnicode_FromString
# 14 PyLong_FromLongLong  15 PyErr_SetString  16 ValueError
# 17 PySequence_GetItem  18 validate_raw
_iface = (ctypes.c_void_p * 19)(
    id(None),                                  # 0: reserved (unused)
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyObject_IsInstance'),
    _address(_python, 'PyList_Type'),
    _address(_python, 'PyLong_Type'),
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'PySet_New'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyUnicode_FromString'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyErr_SetString'),
    id(ValueError),
    _address(_python, 'PySequence_GetItem'),
    id(_HOLD_VALIDATE_RAW),
)
_lib.dd_init(ctypes.byref(_iface))


def options(prompt, candidates, *, max_length=None):
    """Direct next-token candidate options — logic in assembly."""
    return _lib.dd_options(prompt, candidates, max_length)

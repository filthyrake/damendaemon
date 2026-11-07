"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 9: query_rows logic lives in
native/attention_budget_asm.s (exact-int type check, positivity check,
per-row budget arithmetic, unfulfillable-budget and floor-division
paths — all via PyNumber_* so arbitrary-precision semantics are
preserved); this bridge only compiles/caches the dylib, resolves the
interface table, and binds the entry point.
"""
import ctypes
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'attention_budget_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libattention_budget_asm.dylib')
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
_lib.dd_query_rows.restype = ctypes.py_object
_lib.dd_query_rows.argtypes = [ctypes.py_object] * 5

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/attention_budget_asm.s (offsets/8):
#  0 PyLong_Type   1 PyLong_FromLongLong  2 PyNumber_Add
#  3 PyNumber_Multiply  4 PyNumber_FloorDivide  5 PyObject_RichCompareBool
#  6 Py_DecRef     7 ValueError  8 PyErr_SetString
_iface = (ctypes.c_void_p * 9)(
    _address(_python, 'PyLong_Type'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyNumber_Add'),
    _address(_python, 'PyNumber_Multiply'),
    _address(_python, 'PyNumber_FloorDivide'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'Py_DecRef'),
    id(ValueError),
    _address(_python, 'PyErr_SetString'),
)
_lib.dd_init(ctypes.byref(_iface))


def query_rows(heads, dim, splits, scratch_bytes, output_bytes=4):
    """Attention scratch budget row bound — logic in assembly."""
    return _lib.dd_query_rows(heads, dim, splits, scratch_bytes, output_bytes)

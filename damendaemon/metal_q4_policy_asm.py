"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 11: validate logic lives in
native/metal_q4_policy_asm.s (the 8-entry Qwen2.5-0.5B geometry dict,
the any(config.get(key) != value ...) scan with short-circuit, the
rope_scaling/context/max_batch gates with exact-int type checks and
chained comparisons — all via the C API so arbitrary-precision and
exception semantics are preserved); this bridge only compiles/caches
the dylib, resolves the interface table, and binds the entry point.
"""
import ctypes
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'metal_q4_policy_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libmetal_q4_policy_asm.dylib')
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
_lib.dd_validate.restype = ctypes.py_object
_lib.dd_validate.argtypes = [ctypes.py_object] * 3

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/metal_q4_policy_asm.s (offsets/8):
#  0 GetAttrString  1 PyObject_Call  2 PyDict_New  3 PyDict_SetItem
#  4 PyUnicode_FromString  5 PyLong_FromLongLong  6 PyLong_Type
#  7 PyObject_RichCompareBool  8 PyObject_IsTrue  9 Py_DecRef
# 10 PyTuple_New  11 PyTuple_SetItem  12 PyObject_GetIter
# 13 PyIter_Next  14 PySequence_GetItem  15 None
# 16 PyErr_SetString  17 ValueError
_iface = (ctypes.c_void_p * 18)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyUnicode_FromString'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyLong_Type'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'PyObject_GetIter'),
    _address(_python, 'PyIter_Next'),
    _address(_python, 'PySequence_GetItem'),
    id(None),
    _address(_python, 'PyErr_SetString'),
    id(ValueError),
)
_lib.dd_init(ctypes.byref(_iface))


def validate(config, context, max_batch=None):
    """Metal Q4 admission policy — logic in assembly."""
    return _lib.dd_validate(config, context, max_batch)

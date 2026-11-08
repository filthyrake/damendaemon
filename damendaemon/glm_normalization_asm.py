"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 31: rms_norm logic lives in
native/glm_normalization_asm.s (square/mean(-1,keepdim=True),
isfinite/all gates, left-associative multiply chain, RuntimeError);
this bridge only compiles/caches the dylib, resolves the interface
table, and binds the entry point.
"""
import ctypes
import os
import subprocess
import sys

import torch

_HOLD = [torch, torch.isfinite, torch.rsqrt, True, RuntimeError]


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'glm_normalization_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libglm_normalization_asm.dylib')
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
_lib.dd_rms_norm.restype = ctypes.py_object
_lib.dd_rms_norm.argtypes = [ctypes.py_object] * 3

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/glm_normalization_asm.s (offsets/8):
#  0 GetAttrString  1 Call  2 TupleNew  3 TupleSetItem  4 DecRef  5 DictNew
#  6 DictSetItem  7 FromLongLong  8 IsTrue  9 RuntimeError 10 SetString
# 11 NumAdd  12 torch.isfinite  13 torch.rsqrt  14 True  15 NumMultiply
# 16 FromString
_iface = (ctypes.c_void_p * 17)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyObject_IsTrue'),
    id(RuntimeError),
    _address(_python, 'PyErr_SetString'),
    _address(_python, 'PyNumber_Add'),
    id(torch.isfinite),
    id(torch.rsqrt),
    id(True),
    _address(_python, 'PyNumber_Multiply'),
    _address(_python, 'PyUnicode_FromString'),
)
_lib.dd_init(ctypes.byref(_iface))


def rms_norm(value, weight, epsilon):
    """Checked GLM RMS normalization — logic in assembly."""
    return _lib.dd_rms_norm(value, weight, epsilon)

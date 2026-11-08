"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 30: finite logic lives in
native/reference_numerics_asm.s (torch.isfinite(value).all() with the
f-string message via PyUnicode_Concat); this bridge only
compiles/caches the dylib, resolves the interface table, and binds
the entry point.
"""
import ctypes
import os
import subprocess
import sys

import torch

_HOLD_TORCH = torch


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'reference_numerics_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libreference_numerics_asm.dylib')
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
_lib.dd_finite.restype = ctypes.py_object
_lib.dd_finite.argtypes = [ctypes.py_object] * 2

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

_PREFIX = 'nonfinite reference '

# ABI table shared with native/reference_numerics_asm.s (offsets/8):
#  0 GetAttrString  1 PyObject_Call  2 PyTuple_New  3 PyTuple_SetItem
#  4 Py_DecRef  5 PyObject_IsTrue  6 PyUnicode_Concat  7 ValueError
#  8 PyErr_SetString  9 PyUnicode_AsUTF8AndSize  10 torch.isfinite
#  11 prefix str
_iface = (ctypes.c_void_p * 12)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyUnicode_Concat'),
    id(ValueError),
    _address(_python, 'PyErr_SetString'),
    _address(_python, 'PyUnicode_AsUTF8AndSize'),
    id(torch.isfinite),
    id(_PREFIX),
)
_lib.dd_init(ctypes.byref(_iface))


def finite(value, stage):
    """Fail-closed nonfinite gate — logic in assembly."""
    return _lib.dd_finite(value, stage)

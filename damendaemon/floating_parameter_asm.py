"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 29: require_floating logic lives in
native/floating_parameter_asm.s (dtype membership via
PySequence_Contains over the four dtype objects, ValueError with the
concatenated message); this bridge only compiles/caches the dylib,
resolves the interface table, and binds the entry point.
"""
import ctypes
import os
import subprocess
import sys

import torch

_HOLD_TORCH = torch  # iface lifetime


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'floating_parameter_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libfloating_parameter_asm.dylib')
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
_lib.dd_require_floating.restype = ctypes.py_object
_lib.dd_require_floating.argtypes = [ctypes.py_object] * 2

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_PREFIX = 'unsupported floating checkpoint parameter dtype: '

# ABI table shared with native/floating_parameter_asm.s (offsets/8):
#  0 GetAttrString  1 PyObject_Call  2 PyTuple_New  3 PyTuple_SetItem
#  4 Py_DecRef  5 PySequence_Contains  6 ValueError  7 PyErr_SetString
#  8 dtype tuple  9 prefix str  14 PyUnicode_Concat  15 PyUnicode_AsUTF8AndSize
_iface = (ctypes.c_void_p * 16)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PySequence_Contains'),
    id(ValueError),
    _address(_python, 'PyErr_SetString'),
    id(_DTYPES),
    id(_PREFIX),
    id(None),
    id(None),
    id(None),
    id(None),
    _address(_python, 'PyUnicode_Concat'),
    _address(_python, 'PyUnicode_AsUTF8AndSize'),
)
_lib.dd_init(ctypes.byref(_iface))


def require_floating(value, name):
    """Floating-dtype gate — logic in assembly."""
    return _lib.dd_require_floating(value, name)

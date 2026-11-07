"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 10: use_packed_projection logic lives in
native/metal_projection_policy_asm.s (row-count fast path, fp32 gate,
max(products) >= 2**32 unsigned-index bound, and the 4M-element packed
threshold — all via PyNumber_*/PyObject_RichCompareBool so arbitrary
precision and exact 2**32 boundary behavior are preserved); this bridge
only compiles/caches the dylib, resolves the interface table, and binds
the entry point.
"""
import ctypes
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'metal_projection_policy_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libmetal_projection_policy_asm.dylib')
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
_lib.dd_use_packed_projection.restype = ctypes.py_object
_lib.dd_use_packed_projection.argtypes = [ctypes.py_object] * 4

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/metal_projection_policy_asm.s (offsets/8):
#  0 PyObject_IsTrue  1 PyLong_FromLongLong  2 PyNumber_Multiply
#  3 PyObject_RichCompareBool  4 Py_DecRef  5 True  6 False
_iface = (ctypes.c_void_p * 7)(
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyNumber_Multiply'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'Py_DecRef'),
    id(True),
    id(False),
)
_lib.dd_init(ctypes.byref(_iface))


def use_packed_projection(rows, outputs, width, fp32):
    """Packed-projection dispatch policy — logic in assembly."""
    return _lib.dd_use_packed_projection(rows, outputs, width, fp32)

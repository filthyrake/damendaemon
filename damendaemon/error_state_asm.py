"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 14: detached_error logic lives in
native/error_state_asm.s (BaseExceptionGroup dispatch with recursive
derive, copy.copy + with_traceback(None), the try/except-Exception
realized via PyErr_Occurred/PyErr_ExceptionMatches/PyErr_Clear with
RuntimeError(str(error)) replacement, and the __cause__/__context__/
__notes__ snapshot writes); this bridge only compiles/caches the
dylib, resolves the interface table, and binds the entry point.
"""
import copy
import ctypes
import os
import subprocess
import sys

_HOLD_COPY = copy.copy  # keep the iface entry alive


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'error_state_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'liberror_state_asm.dylib')
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
_lib.dd_detached_error.restype = ctypes.py_object
_lib.dd_detached_error.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/error_state_asm.s (offsets/8):
#  0 GetAttrString  1 SetAttrString  2 PyObject_Call  3 PyTuple_New
#  4 PyTuple_SetItem  5 Py_DecRef  6 PyObject_IsInstance  7 PyObject_Size
#  8 PySequence_GetItem  9 PyList_New  10 PyList_Append
# 11 PyErr_Occurred  12 PyErr_ExceptionMatches  13 PyErr_Clear
# 14 copy.copy  15 BaseExceptionGroup  16 RuntimeError  17 Exception
# 18 str  19 None  20 list
_iface = (ctypes.c_void_p * 21)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_SetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyObject_IsInstance'),
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PySequence_GetItem'),
    _address(_python, 'PyList_New'),
    _address(_python, 'PyList_Append'),
    _address(_python, 'PyErr_Occurred'),
    _address(_python, 'PyErr_ExceptionMatches'),
    _address(_python, 'PyErr_Clear'),
    id(_HOLD_COPY),
    id(BaseExceptionGroup),
    id(RuntimeError),
    id(Exception),
    id(str),
    id(None),
    id(list),
)
_lib.dd_init(ctypes.byref(_iface))


def detached_error(error):
    """Detached exception snapshot — logic in assembly."""
    return _lib.dd_detached_error(error)

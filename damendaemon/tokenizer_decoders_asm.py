"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 2: all logic lives in
native/tokenizer_decoders_asm.s; this bridge only compiles/caches the
dylib, resolves the C API + singletons into the interface table, and
binds the two entry points. GIL held via PyDLL (the asm calls back).
"""
import ctypes
import json
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'tokenizer_decoders_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libtokenizer_decoders_asm.dylib')
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
_lib.dd_decoder_steps.restype = ctypes.py_object
_lib.dd_decoder_steps.argtypes = [ctypes.py_object]
_lib.dd_is_bytelevel.restype = ctypes.py_object
_lib.dd_is_bytelevel.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/tokenizer_decoders_asm.s (offsets/8):
#  0 json.loads      8 PyObject_GetAttrString  16 PyObject_Call
# 24 PyList_New     32 PyList_Append          40 PyList_GetItem
# 48 PyList_SetSlice 56 PyTuple_New           64 PyTuple_SetItem
# 72 PyDict_GetItemWithError  80 PyUnicode_CompareWithASCIIString
# 88 PyObject_GetItem  96 Py_DecRef           104 None
# 112 'type'        120 'decoders'            128 True  136 False
_iface = (ctypes.c_void_p * 18)(
    id(json.loads),
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyList_New'),
    _address(_python, 'PyList_Append'),
    _address(_python, 'PyList_GetItem'),
    _address(_python, 'PyList_SetSlice'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'PyDict_GetItemWithError'),
    _address(_python, 'PyUnicode_CompareWithASCIIString'),
    _address(_python, 'PyObject_GetItem'),
    _address(_python, 'Py_DecRef'),
    id(None),
    id('type'),
    id('decoders'),
    id(True),
    id(False),
)
_lib.dd_init(ctypes.byref(_iface))


def decoder_steps(decoder):
    """Flatten a tokenizer decoder into its step dicts — logic in assembly."""
    return _lib.dd_decoder_steps(decoder)


def is_bytelevel(decoder):
    """True only for one ByteLevel step, allowing transparent Sequence wrappers."""
    return _lib.dd_is_bytelevel(decoder)

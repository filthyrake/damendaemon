"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 6: the entire module — the BYTE_VALUES
alphabet, ByteLevelStream construction/validation, feed (suppression,
added-token dispatch, special/literal passthrough, per-char alphabet
mapping with whole-piece UTF-8 fallback), and finish — lives in
native/bytelevel_stream_asm.s. This bridge only compiles/caches the
dylib, resolves the 39-slot interface table, and binds entry points.
"""
import codecs
import ctypes
import os
import subprocess
import sys

from .tokenizer_decoders import is_bytelevel


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'bytelevel_stream_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libbytelevel_stream_asm.dylib')
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
_lib.dd_byte_values.restype = ctypes.py_object
_lib.dd_byte_values.argtypes = []
_lib.dd_stream_init.restype = ctypes.py_object
_lib.dd_stream_init.argtypes = [ctypes.py_object] * 3
_lib.dd_feed.restype = ctypes.py_object
_lib.dd_feed.argtypes = [ctypes.py_object] * 2
_lib.dd_finish.restype = ctypes.py_object
_lib.dd_finish.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# Held references: id() of temporaries (dict literals, possibly-freed
# string constants) dangle once the object is collected and the address
# is reused — the asm then reads unrelated objects as live ones.
_S_UTF8 = 'utf-8'
_S_REPLACE = 'replace'
_S_ERRORS = 'errors'
_S_FINAL = 'final'
_EMPTY_DICT = {}

# ABI table shared with native/bytelevel_stream_asm.s (offsets/8):
#  0 GetAttrString   8 SetAttrString   16 PyObject_Call
# 24 PyTuple_New   32 PyTuple_SetItem  40 Py_DecRef
# 48 PyDict_New    56 PyDict_SetItem   64 PyDict_GetItemWithError
# 72 PyDict_Contains  80 PySequence_Contains  88 PyObject_IsTrue
# 96 PyErr_Clear  104 PyErr_SetString  112 ValueError
# 120 None  128 PyUnicode_FromOrdinal  136 PyUnicode_AsUTF8String
# 144 PyBytes_FromStringAndSize  152 PyLong_AsLongLong  160 PyUnicode_Concat
# 168 str  176 PyUnicode_GetLength  184 PyUnicode_ReadChar
# 192 PyLong_FromLongLong  200 codecs.getincrementaldecoder  208 frozenset
# 216 is_bytelevel  224 'utf-8'  232 'replace'  240 'errors'
# 248 'final'  256 True  264 ''  272 ()  280 {}  288 False
# 296 PyDict_SetItemString  304 PyObject_IsInstance
_iface = (ctypes.c_void_p * 39)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_SetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyDict_GetItemWithError'),
    _address(_python, 'PyDict_Contains'),
    _address(_python, 'PySequence_Contains'),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyErr_Clear'),
    _address(_python, 'PyErr_SetString'),
    id(ValueError),
    id(None),
    _address(_python, 'PyUnicode_FromOrdinal'),
    _address(_python, 'PyUnicode_AsUTF8String'),
    _address(_python, 'PyBytes_FromStringAndSize'),
    _address(_python, 'PyLong_AsLongLong'),
    _address(_python, 'PyUnicode_Concat'),
    id(str),
    _address(_python, 'PyUnicode_GetLength'),
    _address(_python, 'PyUnicode_ReadChar'),
    _address(_python, 'PyLong_FromLongLong'),
    id(codecs.getincrementaldecoder),
    id(frozenset),
    id(is_bytelevel),
    id(_S_UTF8),
    id(_S_REPLACE),
    id(_S_ERRORS),
    id(_S_FINAL),
    id(True),
    id(''),
    id(()),
    id(_EMPTY_DICT),
    id(False),
    _address(_python, 'PyDict_SetItemString'),
    _address(_python, 'PyObject_IsInstance'),
)
_lib.dd_init(ctypes.byref(_iface))

BYTE_VALUES = _lib.dd_byte_values()


class ByteLevelStream:
    """Incremental CPU tokenizer ByteLevel decoding — logic in assembly."""

    def __init__(self, tokenizer, allowed_added):
        _lib.dd_stream_init(self, tokenizer, allowed_added)

    def feed(self, token):
        return _lib.dd_feed(self, token)

    def finish(self):
        return _lib.dd_finish(self)

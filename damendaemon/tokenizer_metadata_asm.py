"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 7: the entire module — bounded metadata
readers (with-protocol file handling, byte limits, parse_metadata_json
dispatch), special_token_text, named-template selection, and
checkpoint_template — lives in native/tokenizer_metadata_asm.s. The
limits are resolved at call time from the SHIM module's globals, so
test-suite patching of MAX_CHECKPOINT_TEMPLATE_BYTES works unchanged.
"""
import ctypes
import os
import subprocess
import sys

from .safetensor_header import parse_metadata_json
from pathlib import Path


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'tokenizer_metadata_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libtokenizer_metadata_asm.dylib')
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
_lib.dd_read_object.restype = ctypes.py_object
_lib.dd_read_object.argtypes = [ctypes.py_object] * 2
_lib.dd_read_tokenizer_config.restype = ctypes.py_object
_lib.dd_read_tokenizer_config.argtypes = [ctypes.py_object]
_lib.dd_read_generation_config.restype = ctypes.py_object
_lib.dd_read_generation_config.argtypes = [ctypes.py_object]
_lib.dd_special_token_text.restype = ctypes.py_object
_lib.dd_special_token_text.argtypes = [ctypes.py_object]
_lib.dd_checkpoint_template.restype = ctypes.py_object
_lib.dd_checkpoint_template.argtypes = [ctypes.py_object] * 3

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# Held references (module globals): their id()s go into the iface table
# and must outlive the asm (id() of a temporary dangles).
_S_UTF8 = 'utf-8'
_S_NAME = 'name'
_S_TEMPLATE = 'template'
_S_DEFAULT = 'default'
_S_TOOL_USE = 'tool_use'
_S_CONTENT = 'content'
_S_CHAT_TEMPLATE = 'chat_template'
_S_BOS = 'bos_token'
_S_EOS = 'eos_token'
# The shim module itself (resolves the patchable MAX_* globals at call
# time). During the shim's own import it is already in sys.modules with
# the constants defined above the bridge import.
_shim = sys.modules[(__package__ or '') + '.tokenizer_metadata']
assert _shim is not None

# ABI table shared with native/tokenizer_metadata_asm.s (offsets/8):
#  0 GetAttrString    8 PyObject_Call  16 PyTuple_New  32 PyTuple_SetItem
# 40 Py_DecRef       48 PyDict_New    56 PyDict_SetItem 64 PyDict_GetItemWithError
# 72 PyDict_Contains 88 PyObject_IsTrue 96 PyErr_Clear 104 PyErr_SetString
#112 ValueError     120 None         136 PyUnicode_AsEncodedString
#144 PyUnicode_FromString 160 PyUnicode_Concat 168 dict 176 list 184 str
#192 PyObject_Size  200 PyIter_Next  208 PyObject_GetIter 216 PySequence_GetItem
#224 PyLong_AsLongLong 232 PyLong_FromLongLong 240 PyBytes_Size
#248 parse_metadata_json 256 Path 264 shim module 272 True
#280 PyObject_IsInstance 288 'name' 296 'template' 304 'default'
#312 'tool_use' 320 'content' 328 'chat_template' 336 'utf-8'
#344 PyErr_SetObject 352 'bos_token' 360 'eos_token' 368 PyUnicode_AsUTF8AndSize
_iface = (ctypes.c_void_p * 47)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    ctypes.c_void_p(0),  # 32 not used via this slot; see SetItem below
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyDict_GetItemWithError'),
    _address(_python, 'PyDict_Contains'),
    ctypes.c_void_p(0),  # 80 unused
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyErr_Clear'),
    _address(_python, 'PyErr_SetString'),
    id(ValueError),
    id(None),
    ctypes.c_void_p(0),  # 128 unused
    _address(_python, 'PyUnicode_AsEncodedString'),
    _address(_python, 'PyUnicode_FromString'),
    ctypes.c_void_p(0),  # 152 unused
    _address(_python, 'PyUnicode_Concat'),
    id(dict),
    id(list),
    id(str),
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PyIter_Next'),
    _address(_python, 'PyObject_GetIter'),
    _address(_python, 'PySequence_GetItem'),
    _address(_python, 'PyLong_AsLongLong'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyBytes_Size'),
    id(parse_metadata_json),
    id(Path),
    id(_shim),
    id(True),
    _address(_python, 'PyObject_IsInstance'),
    id(_S_NAME),
    id(_S_TEMPLATE),
    id(_S_DEFAULT),
    id(_S_TOOL_USE),
    id(_S_CONTENT),
    id(_S_CHAT_TEMPLATE),
    id(_S_UTF8),
    _address(_python, 'PyErr_SetObject'),
    id(_S_BOS),
    id(_S_EOS),
    _address(_python, 'PyUnicode_AsUTF8AndSize'),
)
_lib.dd_init(ctypes.byref(_iface))


def _read_object(path, limit):
    return _lib.dd_read_object(path, limit)


def read_tokenizer_config(path):
    return _lib.dd_read_tokenizer_config(path)


def read_generation_config(path):
    return _lib.dd_read_generation_config(path)


def special_token_text(metadata):
    return _lib.dd_special_token_text(metadata)


def checkpoint_template(path, metadata, *, tools=False):
    return _lib.dd_checkpoint_template(path, metadata, tools)

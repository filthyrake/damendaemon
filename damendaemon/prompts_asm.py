"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 3: all prompts.py logic lives in
native/prompts_asm.s; this bridge only compiles/caches the dylib,
resolves the C API into the interface table, and binds the five
entry points. GIL held via PyDLL.
"""
import ctypes
import os
import subprocess
import sys


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'prompts_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libprompts_asm.dylib')
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
for _name, _args in (('dd_validate_lengths', 3), ('dd_decoder_prefix', 2),
                     ('dd_validate_raw', 2), ('dd_decode_output', 2)):
    getattr(_lib, _name).restype = ctypes.py_object
    getattr(_lib, _name).argtypes = [ctypes.py_object] * _args
_lib.dd_encode.restype = ctypes.py_object
_lib.dd_encode.argtypes = [ctypes.py_object] * 7

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/prompts_asm.s (offsets/8):
#  0 GetAttrString  8 PyObject_Call  16 PyObject_IsTrue
# 24 PyLong_AsLongLong  32 PyDict_New  40 PyDict_SetItemString
# 48 PyObject_GetItem  56 PyErr_Clear  64 PyErr_SetString
# 72 PyExc_ValueError  80 None  88 Py_True  96 &PyUnicode_Type
# 104 &PyLong_Type  112 &PyList_Type  120 Py_DecRef  128 PyTuple_New
# 136 PyTuple_SetItem  144 PyUnicode_AsUTF8String  152 PyList_GetSlice
# 160 PyUnicode_CompareWithASCIIString  168 'decoder_start_token_id' key
# 176 PyList_New  184 PyList_SetItem
_iface = (ctypes.c_void_p * 24)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyLong_AsLongLong'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItemString'),
    _address(_python, 'PyObject_GetItem'),
    _address(_python, 'PyErr_Clear'),
    _address(_python, 'PyErr_SetString'),
    id(ValueError),
    id(None),
    id(True),
    id(str),
    id(int),
    id(list),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'PyUnicode_AsUTF8String'),
    _address(_python, 'PyList_GetSlice'),
    _address(_python, 'PyUnicode_CompareWithASCIIString'),
    id('decoder_start_token_id'),
    _address(_python, 'PyList_New'),
    _address(_python, 'PyList_SetItem'),
)
_lib.dd_init(ctypes.byref(_iface))


def validate_lengths(executor, prompt_tokens, max_tokens):
    return _lib.dd_validate_lengths(executor, prompt_tokens, max_tokens)


def decoder_prefix(executor, prompt):
    """Repetition penalties apply to decoder history, not encoder input."""
    return _lib.dd_decoder_prefix(executor, prompt)


def validate_raw(prompt, *, max_length=None):
    return _lib.dd_validate_raw(prompt, max_length)


def encode(model, messages, raw_prompt=None, enable_thinking=None, tools=None,
           tool_json=False, reasoning_effort=None):
    return _lib.dd_encode(model, messages, raw_prompt, enable_thinking,
                          tools, tool_json, reasoning_effort)


def decode_output(model, tokens):
    """Generated continuations may preserve space that full-text decoding strips."""
    return _lib.dd_decode_output(model, tokens)

"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 39 — class-def module with a regular method
body: the class statement (full __build_class__ semantics) and the
tool_chat_tokens body (render_tool_chat call, tokenizer.encode with
add_special_tokens=False, the not-tokens / len>MAX_TOKENS bound gates)
live in native/glm_tool_text_asm.s. This bridge only compiles/caches
the dylib, resolves the interface table, supplies the logic-free
method protocol shell, and binds the resulting class. __init__ is
inherited from GlmText via real class machinery.
"""
import ctypes
import os
import subprocess
import sys

from .glm_text import GlmText, MAX_TOKENS
from .glm_flash_tool_text import render_tool_chat

_HOLD = (GlmText, MAX_TOKENS, render_tool_chat)


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'glm_tool_text_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libglm_tool_text_asm.dylib')
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
_lib.dd_classes.restype = ctypes.py_object
_lib.dd_classes.argtypes = [ctypes.py_object]
_lib.dd_tool_chat_tokens.restype = ctypes.py_object
_lib.dd_tool_chat_tokens.argtypes = [ctypes.py_object] * 4

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _tool_chat_tokens(self, messages, tools, **options):
    """Protocol shell — logic in native/glm_tool_text_asm.s."""
    return _lib.dd_tool_chat_tokens(self, messages, tools, options)


# ABI table shared with native/glm_tool_text_asm.s (offsets/8):
#  0 GetAttrString  1 Call  2 TupleNew  3 TupleSetItem  4 DecRef  5 DictNew
#  6 DictSetItem  7 FromString  8 TypeIsSubtype  9 Size  10 SeqGetItem
# 11 SetErr  12 ErrClear  13 type  14 GlmText  15 MAX_TOKENS
# 16 render_tool_chat  17 ValueError  18 False  19 _tool_chat_tokens
# 20 IsTrue  21 RichCompareBool  22 FromSsize_t  23 TypeError
_iface = (ctypes.c_void_p * 24)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyDict_New'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyUnicode_FromString'),
    _address(_python, 'PyType_IsSubtype'),
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PySequence_GetItem'),
    _address(_python, 'PyErr_SetString'),
    _address(_python, 'PyErr_Clear'),
    id(type),
    id(GlmText),
    id(MAX_TOKENS),
    id(render_tool_chat),
    id(ValueError),
    id(False),
    id(_tool_chat_tokens),
    _address(_python, 'PyObject_IsTrue'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'PyLong_FromSsize_t'),
    id(TypeError),
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
GlmToolText = _classes[0]

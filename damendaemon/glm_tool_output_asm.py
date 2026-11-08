"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 40 — class-def with __init__ AND close method
bodies: the class statement (full __build_class__ semantics) and both
method bodies (the isinstance GlmToolText gate, real super().__init__,
the CONTROLS+wire tuple concat and ByteLevelStream/FlashToolOutput
construction, self.parser.close()) live in
native/glm_tool_output_asm.s. This bridge only compiles/caches the
dylib, resolves the interface table, supplies logic-free protocol
shells, builds the class, and fills the class registry the asm uses
for super() resolution.
"""
import ctypes
import os
import subprocess
import sys

from .glm_generation import TokenStream
from .glm_tool_text import GlmToolText
from .glm_flash_tool_output import FlashToolOutput
from .glm_flash_tools import OPEN, CLOSE, KEY, END_KEY, VALUE, END_VALUE
from .glm_text import CONTROLS
from .bytelevel_stream import ByteLevelStream

_HOLD = (TokenStream, GlmToolText, FlashToolOutput, OPEN, CLOSE, KEY,
         END_KEY, VALUE, END_VALUE, CONTROLS, ByteLevelStream)

_CLSREG = {}
_KEY = 'GlmToolTokenStream'


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'glm_tool_output_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libglm_tool_output_asm.dylib')
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
_lib.dd_tool_stream_init.restype = ctypes.py_object
_lib.dd_tool_stream_init.argtypes = [ctypes.py_object] * 3
_lib.dd_tool_stream_close.restype = ctypes.py_object
_lib.dd_tool_stream_close.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _init(self, text, codec):
    """Protocol shell — logic in native/glm_tool_output_asm.s."""
    return _lib.dd_tool_stream_init(self, text, codec)


def _close(self):
    """Protocol shell — logic in native/glm_tool_output_asm.s."""
    return _lib.dd_tool_stream_close(self)


# ABI table shared with native/glm_tool_output_asm.s (offsets/8):
#  0 GetAttrString  1 Call  2 TupleNew  3 TupleSetItem  4 DecRef  5 DictNew
#  6 DictSetItem  7 FromString  8 TypeIsSubtype  9 Size  10 SeqGetItem
# 11 SetErr  12 ErrClear  13 type  14 TokenStream  15 GlmToolText
# 16 ByteLevelStream  17 FlashToolOutput  18 CONTROLS  19 OPEN  20 CLOSE
# 21 KEY  22 END_KEY  23 VALUE  24 END_VALUE  25 ValueError  26 TypeError
# 27 IsInstance  28 PyNumber_Add  29 SetAttrString  30 super  31 _CLSREG
# 32 key  33 DictGetItemWithError  34 PyErr_Occurred  35 _init  36 _close
# 37 None
_iface = (ctypes.c_void_p * 38)(
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
    id(TokenStream),
    id(GlmToolText),
    id(ByteLevelStream),
    id(FlashToolOutput),
    id(CONTROLS),
    id(OPEN),
    id(CLOSE),
    id(KEY),
    id(END_KEY),
    id(VALUE),
    id(END_VALUE),
    id(ValueError),
    id(TypeError),
    _address(_python, 'PyObject_IsInstance'),
    _address(_python, 'PyNumber_Add'),
    _address(_python, 'PyObject_SetAttrString'),
    id(super),
    id(_CLSREG),
    id(_KEY),
    _address(_python, 'PyDict_GetItemWithError'),
    _address(_python, 'PyErr_Occurred'),
    id(_init),
    id(_close),
    id(None),
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
GlmToolTokenStream = _classes[0]
_CLSREG[_KEY] = GlmToolTokenStream

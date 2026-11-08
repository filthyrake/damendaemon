"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 41 — the tool wire-format output engine:
the two class statements (full __build_class__ semantics; FlashToolOutput
is BASELESS) and ALL seven method bodies live in
native/glm_flash_tool_output_asm.s: __init__ (FlashTools isinstance gate
+ six attribute seeds), _usable (closed/failed gate), close, feed (byte
budget, StringIO buffer fast path, OPEN boundary split, partial-opener
suffix scan, except BaseException -> failed=True + close + bare raise),
finish (bool flag gate, ordered STOPS endswith scan, tool_calls dict
event, finally close), and the token-stream __init__/close pair. The
bridge only compiles/caches the dylib, resolves the interface table,
supplies logic-free protocol shells, builds the classes, and fills the
_CLSREG the asm uses for super()/parser resolution at call time.
"""
import ctypes
import os
import subprocess
import sys
from io import StringIO

from .glm_output import GlmOutput, STOPS
from .glm_text import MAX_TEXT_BYTES, CONTROLS
from .glm_generation import TokenStream
from .bytelevel_stream import ByteLevelStream
from .glm_flash_tools import (FlashTools, OPEN, CLOSE, KEY, END_KEY, VALUE,
                              END_VALUE, _text)
from .glm_flash_tool_text import FlashToolText

_HOLD = (StringIO, GlmOutput, STOPS, MAX_TEXT_BYTES, CONTROLS, TokenStream,
         ByteLevelStream, FlashTools, OPEN, CLOSE, KEY, END_KEY, VALUE,
         END_VALUE, _text, FlashToolText)

_CLSREG = {}
_KEY_FTO = 'FlashToolOutput'
_KEY_FTS = 'FlashToolTokenStream'
_S_CONTENT = 'content'
_S_TOOLCALLS = 'tool_calls'
_S_UTF8 = 'utf-8'
_S_EMPTY = ''
_T_EMPTY = ()
_I_ZERO = 0


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'glm_flash_tool_output_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libglm_flash_tool_output_asm.dylib')
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
_lib.dd_fto_init.restype = ctypes.py_object
_lib.dd_fto_init.argtypes = [ctypes.py_object] * 2
_lib.dd_fto_usable.restype = ctypes.py_object
_lib.dd_fto_usable.argtypes = [ctypes.py_object]
_lib.dd_fto_close.restype = ctypes.py_object
_lib.dd_fto_close.argtypes = [ctypes.py_object]
_lib.dd_fto_feed.restype = ctypes.py_object
_lib.dd_fto_feed.argtypes = [ctypes.py_object] * 2
_lib.dd_fto_finish.restype = ctypes.py_object
_lib.dd_fto_finish.argtypes = [ctypes.py_object] * 2
_lib.dd_ts_init.restype = ctypes.py_object
_lib.dd_ts_init.argtypes = [ctypes.py_object] * 3
_lib.dd_ts_close.restype = ctypes.py_object
_lib.dd_ts_close.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _fto_init(self, codec):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_fto_init(self, codec)


def _usable(self):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_fto_usable(self)


def _fto_close(self):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_fto_close(self)


def _feed(self, text):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_fto_feed(self, text)


def _finish(self, *, truncated=False):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_fto_finish(self, truncated)


def _ts_init(self, text, codec):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_ts_init(self, text, codec)


def _ts_close(self):
    """Protocol shell — logic in native/glm_flash_tool_output_asm.s."""
    return _lib.dd_ts_close(self)


# ABI table shared with native/glm_flash_tool_output_asm.s — byte offset =
# index*8, exactly as listed in the .s header (69 entries, 0..544):
_iface = (ctypes.c_void_p * 69)(
    _address(_python, 'PyObject_GetAttrString'),        # 0
    _address(_python, 'PyObject_Call'),                 # 8
    _address(_python, 'PyTuple_New'),                   # 16
    _address(_python, 'PyTuple_SetItem'),              # 24
    _address(_python, 'Py_DecRef'),                     # 32
    _address(_python, 'PyDict_New'),                   # 40
    _address(_python, 'PyDict_SetItem'),               # 48
    _address(_python, 'PyUnicode_FromString'),         # 56
    _address(_python, 'PyType_IsSubtype'),              # 64
    _address(_python, 'PyObject_Size'),                # 72
    _address(_python, 'PySequence_GetItem'),           # 80
    _address(_python, 'PyErr_SetString'),              # 88
    _address(_python, 'PyErr_Clear'),                  # 96
    id(type),                                          # 104
    id(TokenStream),                                  # 112
    id(FlashTools),                                    # 120
    id(GlmOutput),                                    # 128
    id(StringIO),                                      # 136
    id(MAX_TEXT_BYTES),                                # 144
    id(CONTROLS),                                      # 152
    id(OPEN),                                          # 160
    id(_text),                                         # 168
    id(FlashToolText),                                 # 176
    id(ByteLevelStream),                               # 184
    id(CLOSE),                                         # 192
    id(KEY),                                           # 200
    id(END_KEY),                                       # 208
    id(VALUE),                                         # 216
    id(END_VALUE),                                     # 224
    id(ValueError),                                    # 232
    id(TypeError),                                     # 240
    _address(_python, 'PyObject_IsTrue'),               # 248
    _address(_python, 'PyObject_RichCompareBool'),      # 256
    _address(_python, 'PyLong_AsLongLong'),              # 264
    _address(_python, 'PyUnicode_Substring'),           # 272
    _address(_python, 'PyObject_SetAttrString'),        # 280
    _address(_python, 'PyObject_IsInstance'),           # 288
    _address(_python, 'PyNumber_Add'),                 # 296
    _address(_python, 'PyNumber_InPlaceAdd'),           # 304
    _address(_python, 'PyErr_Fetch'),                  # 312
    _address(_python, 'PyErr_Restore'),                # 320
    _address(_python, 'PyBool_Type'),                   # 328
    id(RuntimeError),                                   # 336
    _address(_python, 'PyList_New'),                   # 344
    _address(_python, 'PyList_SetItem'),               # 352
    id(_S_CONTENT),                                    # 360
    id(_S_TOOLCALLS),                                  # 368
    id(_S_UTF8),                                       # 376
    id(STOPS),                                         # 384
    id(None),                                          # 392
    id(True),                                          # 400
    id(_S_EMPTY),                                      # 408
    id(_T_EMPTY),                                      # 416
    id(_I_ZERO),                                       # 424
    id(_fto_init),                                     # 432
    id(_usable),                                       # 440
    id(_fto_close),                                    # 448
    id(_feed),                                         # 456
    id(_finish),                                        # 464
    id(_ts_init),                                      # 472
    id(_ts_close),                                     # 480
    id(False),                                         # 488
    _address(_python, 'PyErr_Occurred'),               # 496
    _address(_python, 'PyLong_FromSsize_t'),            # 504
    id(_CLSREG),                                       # 512
    id(_KEY_FTO),                                      # 520
    id(_KEY_FTS),                                      # 528
    id(super),                                         # 536
    _address(_python, 'PyDict_GetItemWithError'),       # 544
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
FlashToolOutput = _classes[0]
FlashToolTokenStream = _classes[1]
_CLSREG[_KEY_FTO] = FlashToolOutput
_CLSREG[_KEY_FTS] = FlashToolTokenStream

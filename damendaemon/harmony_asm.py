"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 8: the entire module — the _MARKERS table,
TextStream construction (marker validation against the tokenizer, all
17 state attributes, the declared tool-name set), _decode,
_begin_body (role/channel/format validation, recipients, function
handoff names), _commit_body, the full feed state machine
(start/message/end/return/eot/call markers, header/body accumulation,
header bound, control-token rejection, completed/emitted bookkeeping
through stopping.text_delta, closed latch, 3-tuple return), and
output_stream — lives in native/harmony_asm.s. This bridge only
compiles/caches the dylib, resolves the 44-slot interface table, and
binds entry points.
"""
import ctypes
import json
import os
import subprocess
import sys
import uuid

from .stopping import text_delta
from .tool_calls import _json_object, validate_tools


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'harmony_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libharmony_asm.dylib')
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
_lib.dd_markers.restype = ctypes.py_object
_lib.dd_markers.argtypes = []
_lib.dd_text_init.restype = ctypes.py_object
_lib.dd_text_init.argtypes = [ctypes.py_object] * 4
_lib.dd_feed.restype = ctypes.py_object
_lib.dd_feed.argtypes = [ctypes.py_object] * 3
_lib.dd_decode.restype = ctypes.py_object
_lib.dd_decode.argtypes = [ctypes.py_object] * 2
_lib.dd_output_stream.restype = ctypes.py_object
_lib.dd_output_stream.argtypes = [ctypes.py_object] * 7

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


class TextStream:
    """Token-framed Harmony output parser — logic in assembly."""

    def __init__(self, tokenizer, stops=(), tools=None):
        _lib.dd_text_init(self, tokenizer, stops, tools)

    def _decode(self, tokens):
        return _lib.dd_decode(self, tokens)

    def feed(self, token, final=False):
        return _lib.dd_feed(self, token, final)


def output_stream(model, raw_prompt, stops=(), schema=None, logprobs=None,
                  ignore_eos=False, tools=None):
    return _lib.dd_output_stream(model, raw_prompt, stops, schema, logprobs,
                                 ignore_eos, tools)


# Held references: id() of callables/objects in the table must stay
# alive for the process lifetime; module globals below pin them.
_UUID4 = uuid.uuid4
_DUMPS = json.dumps

# ABI table shared with native/harmony_asm.s (offsets/8):
#  0 GetAttrString    8 SetAttrString   16 PyObject_Call
# 24 PyTuple_New    32 PyTuple_SetItem 40 Py_DecRef
# 48 PyDict_New     56 PyDict_SetItem  64 PyDict_GetItemWithError
# 72 PyDict_Contains 80 PySequence_Contains 88 PyObject_IsTrue
# 96 PyErr_Clear   104 PyErr_SetString 112 ValueError
# 120 None         128 PyLong_AsLongLong 136 PyLong_FromLongLong
# 144 PyUnicode_FromString 152 PyUnicode_Concat
# 160 PyObject_GetIter 168 PyObject_Size 176 PyIter_Next
# 208 PySequence_GetItem 216 PySequence_GetSlice 224 PySequence_Concat
# 232 PyList_New   240 PyList_Append
# 248 PyUnicode_CompareWithASCIIString 256 PyDict_Copy
# 264 PySet_New    272 PySet_Add     280 True    288 False
# 296 text_delta   304 _json_object  312 validate_tools
# 320 uuid.uuid4   328 json.dumps   336 TextStream
# 344 PyUnicode_AsUTF8AndSize
_iface = (ctypes.c_void_p * 44)(
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
    _address(_python, 'PyLong_AsLongLong'),
    _address(_python, 'PyLong_FromLongLong'),
    _address(_python, 'PyUnicode_FromString'),
    _address(_python, 'PyUnicode_Concat'),
    _address(_python, 'PyObject_GetIter'),    # 160
    _address(_python, 'PyObject_Size'),
    _address(_python, 'PyIter_Next'),
    id(None),                                    # 184: unused
    id(None),                                    # 192: unused
    id(None),                                    # 200: unused
    _address(_python, 'PySequence_GetItem'),
    _address(_python, 'PySequence_GetSlice'),
    _address(_python, 'PySequence_Concat'),
    _address(_python, 'PyList_New'),
    _address(_python, 'PyList_Append'),
    _address(_python, 'PyUnicode_CompareWithASCIIString'),
    _address(_python, 'PyDict_Copy'),
    _address(_python, 'PySet_New'),
    _address(_python, 'PySet_Add'),
    id(True),
    id(False),
    id(text_delta),
    id(_json_object),
    id(validate_tools),
    id(_UUID4),
    id(_DUMPS),
    id(TextStream),
    _address(_python, 'PyUnicode_AsUTF8AndSize'),
)
_lib.dd_init(ctypes.byref(_iface))

_MARKERS = _lib.dd_markers()

__all__ = ['TextStream', 'output_stream', '_MARKERS']

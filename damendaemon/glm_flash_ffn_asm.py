"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 34 — first class-def module WITH method
bodies: the class statements (full __build_class__ semantics) and both
__init__ bodies (the _limit gate: exact int/float type check + chained
float bound via torch.finfo, then super(cls, self).__init__ with the
kwargs merge activation_limit + **options) live in
native/glm_flash_ffn_asm.s. This bridge only compiles/caches the
dylib, resolves the interface table, supplies logic-free __init__
protocol shells, builds the classes, and fills the class registry
the asm uses for super() resolution.
"""
import ctypes
import os
import subprocess
import sys

import torch

from .glm_ffn import GlmFeedForward
from .glm_packed_ffn import PackedGlmFeedForward

_HOLD = (torch, torch.finfo, torch.float32, GlmFeedForward, PackedGlmFeedForward)

_CLSREG = {}
_KEY_FLASH = 'FlashFeedForward'
_KEY_PACKED = 'PackedFlashFeedForward'
_AKEY = 'activation_limit'
_INT0 = 0


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'glm_flash_ffn_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libglm_flash_ffn_asm.dylib')
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
_lib.dd_limit.restype = ctypes.py_object
_lib.dd_limit.argtypes = [ctypes.py_object]
_lib.dd_flash_ffn_init.restype = ctypes.py_object
_lib.dd_flash_ffn_init.argtypes = [ctypes.py_object] * 4
_lib.dd_packed_flash_ffn_init.restype = ctypes.py_object
_lib.dd_packed_flash_ffn_init.argtypes = [ctypes.py_object] * 4

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _flash_init(self, weights, *, activation_limit=10., **options):
    """Protocol shell — logic in native/glm_flash_ffn_asm.s."""
    return _lib.dd_flash_ffn_init(self, weights, activation_limit, options)


def _packed_flash_init(self, experts, *, activation_limit=10., **options):
    """Protocol shell — logic in native/glm_flash_ffn_asm.s."""
    return _lib.dd_packed_flash_ffn_init(self, experts, activation_limit, options)


def _limit(value):
    """Protocol shell — logic in native/glm_flash_ffn_asm.s."""
    return _lib.dd_limit(value)


# ABI table shared with native/glm_flash_ffn_asm.s (offsets/8):
#  0 GetAttrString  1 Call  2 TupleNew  3 TupleSetItem  4 DecRef  5 DictNew
#  6 DictSetItem  7 FromString  8 TypeIsSubtype  9 Size  10 SeqGetItem
#  11 SetErr  12 ErrClear  13 type  14 PyLong_Type  15 TypeError
#  16 GlmFeedForward  17 PackedGlmFeedForward  18 ValueError  19 PyFloat_Type
#  20 torch.finfo  21 torch.float32  22 int 0  23 PyDict_Merge  24 super
#  25 _CLSREG  26 key 'FlashFeedForward'  27 key 'PackedFlashFeedForward'
#  28 'activation_limit'  29 _flash_init  30 _packed_flash_init
#  31 RichCompareBool  32 DictGetItemWithError  33 PyErr_Occurred
_iface = (ctypes.c_void_p * 34)(
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
    _address(_python, 'PyLong_Type'),
    id(TypeError),
    id(GlmFeedForward),
    id(PackedGlmFeedForward),
    id(ValueError),
    _address(_python, 'PyFloat_Type'),
    id(torch.finfo),
    id(torch.float32),
    id(_INT0),
    _address(_python, 'PyDict_Merge'),
    id(super),
    id(_CLSREG),
    id(_KEY_FLASH),
    id(_KEY_PACKED),
    id(_AKEY),
    id(_flash_init),
    id(_packed_flash_init),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'PyDict_GetItemWithError'),
    _address(_python, 'PyErr_Occurred'),
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
FlashFeedForward = _classes[0]
PackedFlashFeedForward = _classes[1]
_CLSREG[_KEY_FLASH] = FlashFeedForward
_CLSREG[_KEY_PACKED] = PackedFlashFeedForward

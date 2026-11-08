"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 36 — class-def with __init__ body: the
class statement (full __build_class__ semantics) and the __init__ body
(isinstance(weights, dict) conditional, weights.get('q_proj.weight'),
_device admission gate, kwargs merge, real super()) live in
native/cuda_glm_flash_linear_asm.s. This bridge only compiles/caches
the dylib, resolves the interface table, supplies the logic-free
__init__ signature shell, builds the class, and fills the class
registry the asm uses for super() resolution.
"""
import ctypes
import os
import subprocess
import sys

from .glm_flash_linear import _FlashLinearAttention
from .cuda_glm_flash_primitives import _device

_HOLD = (_FlashLinearAttention, _device)

_CLSREG = {}
_KEY = 'CudaFlashLinearAttention'
_WKEY = 'q_proj.weight'
_DKEY = 'device'


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'cuda_glm_flash_linear_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libcuda_glm_flash_linear_asm.dylib')
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
_lib.dd_cuda_init.restype = ctypes.py_object
_lib.dd_cuda_init.argtypes = [ctypes.py_object] * 3

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _cuda_init(self, weights, **options):
    """Protocol shell — logic in native/cuda_glm_flash_linear_asm.s."""
    return _lib.dd_cuda_init(self, weights, options)


# ABI table shared with native/cuda_glm_flash_linear_asm.s (offsets/8):
#  0 GetAttrString  1 Call  2 TupleNew  3 TupleSetItem  4 DecRef  5 DictNew
#  6 DictSetItem  7 FromString  8 TypeIsSubtype  9 Size  10 SeqGetItem
#  11 SetErr  12 ErrClear  13 type  14 dict  15 TypeError
#  16 _FlashLinearAttention  17 _device  18 'q_proj.weight'  19 None
#  20 'device'  21 _CLSREG  22 key 'CudaFlashLinearAttention'
#  23 _cuda_init  24 DictGetItemWithError  25 super  26 PyDict_Merge
#  27 IsInstance  28 PyErr_Occurred
_iface = (ctypes.c_void_p * 29)(
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
    id(dict),
    id(TypeError),
    id(_FlashLinearAttention),
    id(_device),
    id(_WKEY),
    id(None),
    id(_DKEY),
    id(_CLSREG),
    id(_KEY),
    id(_cuda_init),
    _address(_python, 'PyDict_GetItemWithError'),
    id(super),
    _address(_python, 'PyDict_Merge'),
    _address(_python, 'PyObject_IsInstance'),
    _address(_python, 'PyErr_Occurred'),
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
CudaFlashLinearAttention = _classes[0]
_CLSREG[_KEY] = CudaFlashLinearAttention

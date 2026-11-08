"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 33: use_small_tile/_is_a2 logic lives in
native/q4_prefill_policy_asm.s (chained-comparison gate, tuple
membership via PySequence_Contains, per-call torch.cuda.
get_device_properties resolution, lru_cache realized as a bridge-held
dict with lookup/store in asm); this bridge only compiles/caches the
dylib, resolves the interface table, and binds the entry points.

Known divergence (documented in PORTING_STATUS.md): _is_a2 exposes
cache_clear but not cache_info (unused across the repo/tests).
"""
import ctypes
import os
import subprocess
import sys

import torch

_HOLDER = torch  # iface lifetime

_TILES = ((4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096), (4096, 32000))
_CACHE = {}
_A2_NAME = 'NVIDIA A2'
_CUDA = 'cuda'
_INT32 = 32
_INT64 = 64


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'q4_prefill_policy_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libq4_prefill_policy_asm.dylib')
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
_lib.dd_is_a2.restype = ctypes.py_object
_lib.dd_is_a2.argtypes = [ctypes.py_object]
_lib.dd_cache_clear.restype = None
_lib.dd_use_small_tile.restype = ctypes.py_object
_lib.dd_use_small_tile.argtypes = [ctypes.py_object] * 4

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)

# ABI table shared with native/q4_prefill_policy_asm.s (offsets/8):
#  0 GetAttrString  1 Call  2 TupleNew  3 TupleSetItem  4 DecRef
#  5 DictGetItemWithError  6 DictSetItem  7 RichCompareBool  8 BoolFromLong
#  9 DictClear  10 torch  11 _TILES  12 _CACHE  13 'NVIDIA A2'  14 'cuda'
#  15 int 32  16 int 64  17 False  18 PyErr_Occurred  19 PySequence_Contains
_iface = (ctypes.c_void_p * 20)(
    _address(_python, 'PyObject_GetAttrString'),
    _address(_python, 'PyObject_Call'),
    _address(_python, 'PyTuple_New'),
    _address(_python, 'PyTuple_SetItem'),
    _address(_python, 'Py_DecRef'),
    _address(_python, 'PyDict_GetItemWithError'),
    _address(_python, 'PyDict_SetItem'),
    _address(_python, 'PyObject_RichCompareBool'),
    _address(_python, 'PyBool_FromLong'),
    _address(_python, 'PyDict_Clear'),
    id(torch),
    id(_TILES),
    id(_CACHE),
    id(_A2_NAME),
    id(_CUDA),
    id(_INT32),
    id(_INT64),
    id(False),
    _address(_python, 'PyErr_Occurred'),
    _address(_python, 'PySequence_Contains'),
)
_lib.dd_init(ctypes.byref(_iface))


class _LruCacheShell:
    """lru_cache(maxsize=None) protocol shell — cache logic is in asm."""

    def __call__(self, device):
        return _lib.dd_is_a2(device)

    def cache_clear(self):
        _lib.dd_cache_clear()


_is_a2 = _LruCacheShell()


def use_small_tile(rows, outputs, width, device):
    """Hardware-bounded Q4 tile gate — logic in assembly."""
    return _lib.dd_use_small_tile(rows, outputs, width, device)

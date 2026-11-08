"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 42 — class-def with a class attribute and two
method bodies (_stage_batch chain; cache_bytes_per_row sum-genexp) — all
in native/glm_layer_pipeline_asm.s. This bridge only compiles/caches the
dylib, resolves the interface table, supplies logic-free protocol
shells, and builds the class.
"""
import ctypes
import os
import subprocess
import sys

from .layer_pipeline import CpuLayerPipeline, LayerPipelineState as GlmPipelineState
from .glm_layer_stage import GlmLayerStage

_HOLD = (CpuLayerPipeline, GlmPipelineState, GlmLayerStage)


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'glm_layer_pipeline_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libglm_layer_pipeline_asm.dylib')
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
_lib.dd_stage_batch.restype = ctypes.py_object
_lib.dd_stage_batch.argtypes = [ctypes.py_object] * 2
_lib.dd_cache_bytes.restype = ctypes.py_object
_lib.dd_cache_bytes.argtypes = [ctypes.py_object] * 2

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _stage_batch(self, state):
    """Protocol shell — logic in native/glm_layer_pipeline_asm.s."""
    return _lib.dd_stage_batch(self, state)


def _cache_bytes_per_row(self, capacity):
    """Protocol shell — logic in native/glm_layer_pipeline_asm.s."""
    return _lib.dd_cache_bytes(self, capacity)


# ABI table shared with native/glm_layer_pipeline_asm.s — byte offset =
# index*8, exactly as listed in the .s header (28 entries, 0..216):
_iface = (ctypes.c_void_p * 28)(
    _address(_python, 'PyObject_GetAttrString'),        # 0
    _address(_python, 'PyObject_Call'),                 # 8
    _address(_python, 'PyTuple_New'),                   # 16
    _address(_python, 'PyTuple_SetItem'),               # 24
    _address(_python, 'Py_DecRef'),                      # 32
    _address(_python, 'PyDict_New'),                     # 40
    _address(_python, 'PyDict_SetItem'),                 # 48
    _address(_python, 'PyUnicode_FromString'),          # 56
    _address(_python, 'PyType_IsSubtype'),               # 64
    _address(_python, 'PyObject_Size'),                  # 72
    _address(_python, 'PySequence_GetItem'),             # 80
    _address(_python, 'PyErr_SetString'),                # 88
    _address(_python, 'PyErr_Clear'),                    # 96
    id(type),                                            # 104
    id(CpuLayerPipeline),                                # 112
    id(GlmLayerStage),                                   # 120
    id(None),                                            # 128
    id(0),                                               # 136
    id(4),                                               # 144
    _address(_python, 'PyNumber_Multiply'),              # 152
    _address(_python, 'PyNumber_Add'),                   # 160
    _address(_python, 'PyObject_GetIter'),               # 168
    _address(_python, 'PyIter_Next'),                    # 176
    id(_stage_batch),                                    # 184
    id(_cache_bytes_per_row),                            # 192
    _address(_python, 'PyUnicode_FromString'),            # 200 (__prepare__ str — unused slot)
    id(TypeError),                                       # 208
    _address(_python, 'PyErr_Occurred'),                 # 216
)
_lib.dd_init(ctypes.byref(_iface))

_classes = _lib.dd_classes(__name__.rsplit('_asm', 1)[0])
GlmLayerPipeline = _classes[0]

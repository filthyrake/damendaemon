"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 45 — qualified_scope, use_grouped_prefill and
use_packed_prefill (geometry dict build, and-chains with CPython operand
return semantics, chained compares, all()/sum() as explicit loops) live
in native/moe_prefill_asm.s. This bridge only compiles/caches the dylib,
resolves the interface table, holds constants, and supplies logic-free
protocol shells.
"""
import ctypes
import os
import subprocess
import sys

_OLMOE = 'olmoe'
_DEVICE = 'NVIDIA A2'
_QUANT = 'q8_0'
_CLIP = 'clip_qkv'
_I0, _I1, _I2, _I4, _I8, _I16, _I64, _I128 = 0, 1, 2, 4, 8, 16, 64, 128
_I1024, _I2048, _I50304 = 1024, 2048, 50304

_HOLD = (_OLMOE, _DEVICE, _QUANT, _CLIP, _I0, _I1, _I2, _I4, _I8, _I16,
         _I64, _I128, _I1024, _I2048, _I50304)


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'moe_prefill_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libmoe_prefill_asm.dylib')
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
_lib.dd_qualified.restype = ctypes.py_object
_lib.dd_qualified.argtypes = [ctypes.py_object] * 5
_lib.dd_grouped.restype = ctypes.py_object
_lib.dd_grouped.argtypes = [ctypes.py_object] * 2
_lib.dd_packed.restype = ctypes.py_object
_lib.dd_packed.argtypes = [ctypes.py_object] * 2

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _qualified_scope(config, device_name, quantization, context, max_batch):
    """Protocol shell — logic in native/moe_prefill_asm.s."""
    return _lib.dd_qualified(config, device_name, quantization, context, max_batch)


def _use_grouped_prefill(qualified, tokens):
    """Protocol shell — logic in native/moe_prefill_asm.s."""
    return _lib.dd_grouped(qualified, tokens)


def _use_packed_prefill(qualified, lengths):
    """Protocol shell — logic in native/moe_prefill_asm.s."""
    return _lib.dd_packed(qualified, lengths)


# ABI table shared with native/moe_prefill_asm.s — byte offset =
# index*8, exactly as listed in the .s header (35 entries, 0..272):
_iface = (ctypes.c_void_p * 35)(
    _address(_python, 'PyObject_Call'),                 # 0
    _address(_python, 'Py_DecRef'),                      # 8
    _address(_python, 'PyObject_GetAttrString'),         # 16
    _address(_python, 'PyObject_RichCompare'),           # 24
    _address(_python, 'PyObject_IsTrue'),                # 32
    _address(_python, 'PyDict_New'),                     # 40
    _address(_python, 'PyDict_SetItem'),                 # 48
    _address(_python, 'PyIter_Next'),                    # 56
    _address(_python, 'PyObject_GetIter'),               # 64
    _address(_python, 'PyNumber_Add'),                   # 72
    id(True),                                             # 80
    id(False),                                            # 88
    id(None),                                             # 96
    id(_OLMOE),                                           # 104
    id(_I2048),                                           # 112
    id(_I1024),                                           # 120
    id(_I16),                                             # 128
    id(_I64),                                             # 136
    id(_I8),                                              # 144
    id(_I50304),                                          # 152
    id(_DEVICE),                                          # 160
    id(_QUANT),                                           # 168
    id(_CLIP),                                            # 176
    id(_I1),                                              # 184
    id(_I2),                                              # 192
    id(_I4),                                              # 200
    id(_I0),                                              # 208
    id(_I128),                                            # 216
    _address(_python, 'PyTuple_New'),                    # 224
    _address(_python, 'PyTuple_SetItem'),                # 232
    _address(_python, 'PyUnicode_FromString'),           # 240
    _address(_python, 'PySequence_GetItem'),              # 248
    _address(_python, 'PyObject_Size'),                   # 256
    _address(_python, 'PyLong_FromSsize_t'),              # 264
    _address(_python, 'Py_IncRef'),                       # 272
)
_lib.dd_init(ctypes.byref(_iface))

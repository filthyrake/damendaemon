"""ctypes bridge for the hand-written arm64 assembly port of this module.

asm-rewrite branch, module 44 — manifest_digest (Path join, is_file gate,
sha256 of MANIFEST_NAME + NUL + manifest bytes) lives in
native/checkpoint_manifest_asm.s. This bridge only compiles/caches the
dylib, resolves the interface table, holds Path/MANIFEST_NAME/b'\\0'/
hashlib.sha256, and supplies the logic-free protocol shell.
"""
import ctypes
import os
import subprocess
import sys
from hashlib import sha256 as _sha256
from pathlib import Path as _Path

MANIFEST_NAME = 'crc32.txt'
_B_NUL = b'\0'

_HOLD = (_sha256, _Path, MANIFEST_NAME, _B_NUL)


def _build():
    root = os.path.dirname(os.path.abspath(__file__))
    source = os.path.normpath(os.path.join(root, os.pardir, 'native', 'checkpoint_manifest_asm.s'))
    cache = os.path.join(os.path.expanduser('~/.cache'), 'damendaemon-asm')
    target = os.path.join(cache, 'libcheckpoint_manifest_asm.dylib')
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
_lib.dd_manifest_digest.restype = ctypes.py_object
_lib.dd_manifest_digest.argtypes = [ctypes.py_object]

_framework = os.path.join(sys.base_prefix, 'Python')
_python = ctypes.CDLL(_framework if os.path.exists(_framework) else None)


def _manifest_digest(checkpoint):
    """Protocol shell — logic in native/checkpoint_manifest_asm.s."""
    return _lib.dd_manifest_digest(checkpoint)


# ABI table shared with native/checkpoint_manifest_asm.s — byte offset =
# index*8, exactly as listed in the .s header (13 entries, 0..96):
_iface = (ctypes.c_void_p * 13)(
    _address(_python, 'PyObject_Call'),                 # 0
    _address(_python, 'Py_DecRef'),                      # 8
    _address(_python, 'PyObject_GetAttrString'),         # 16
    _address(_python, 'PyNumber_TrueDivide'),            # 24
    _address(_python, 'PyNumber_Add'),                   # 32
    _address(_python, 'PyObject_IsTrue'),                # 40
    id(None),                                             # 48
    id(_Path),                                            # 56
    id(MANIFEST_NAME),                                    # 64
    id(_B_NUL),                                           # 72
    id(_sha256),                                          # 80
    _address(_python, 'PyTuple_New'),                    # 88
    _address(_python, 'PyTuple_SetItem'),                # 96
)
_lib.dd_init(ctypes.byref(_iface))
